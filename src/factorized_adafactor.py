from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class FactorizedAdafactor(Optimizer):
    """Chunked factored second-moment optimizer for large FP32 matrices.

    Unlike AdamW, it does not allocate two full-size moment tensors for every
    parameter. Matrix moments are factorized into one row vector and one column
    vector; 1-D parameters use a full second-moment vector. The update is applied
    in row chunks, so no full-size preconditioner or update tensor is created.

    This implementation intentionally omits first-moment momentum. It is meant
    for a low-memory continuation from a merged checkpoint whose original
    per-source optimizer states cannot be combined into one valid state.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 5e-6,
        beta2_decay: float = -0.8,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        chunk_rows: int = 64,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if beta2_decay >= 0:
            raise ValueError("beta2_decay must be negative")
        if eps <= 0 or clip_threshold <= 0 or chunk_rows <= 0:
            raise ValueError("eps, clip_threshold and chunk_rows must be positive")
        defaults = dict(
            lr=float(lr),
            beta2_decay=float(beta2_decay),
            eps=float(eps),
            clip_threshold=float(clip_threshold),
            weight_decay=float(weight_decay),
            chunk_rows=int(chunk_rows),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            clip_threshold = group["clip_threshold"]
            weight_decay = group["weight_decay"]
            chunk_rows = group["chunk_rows"]
            decay_power = group["beta2_decay"]

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if parameter.dtype != torch.float32 or grad.dtype != torch.float32:
                    raise TypeError(
                        "FactorizedAdafactor continuation currently requires FP32 "
                        "parameters and gradients"
                    )
                if not parameter.is_contiguous() or not grad.is_contiguous():
                    raise ValueError("parameters and gradients must be contiguous")

                state = self.state[parameter]
                step_number = int(state.get("step", 0)) + 1
                state["step"] = step_number
                beta2 = 1.0 - step_number ** decay_power

                if parameter.ndim >= 2:
                    rows = parameter.numel() // parameter.shape[-1]
                    cols = parameter.shape[-1]
                    if rows < 1 or cols < 1:
                        raise ValueError("invalid matrix shape")

                    if "row_sq" not in state:
                        state["row_sq"] = torch.zeros(
                            rows, dtype=torch.float32, device=parameter.device
                        )
                        state["col_sq"] = torch.zeros(
                            cols, dtype=torch.float32, device=parameter.device
                        )

                    row_sq = state["row_sq"]
                    col_sq = state["col_sq"]
                    if row_sq.numel() != rows or col_sq.numel() != cols:
                        raise ValueError("optimizer state shape does not match parameter")

                    p2 = parameter.view(rows, cols)
                    g2 = grad.view(rows, cols)
                    col_accum = torch.zeros_like(col_sq)

                    # Pass 1: update factored moments using small temporary chunks.
                    for start in range(0, rows, chunk_rows):
                        end = min(start + chunk_rows, rows)
                        block_sq = g2[start:end].square()
                        row_mean = block_sq.mean(dim=1)
                        row_sq[start:end].mul_(beta2).add_(
                            row_mean, alpha=1.0 - beta2
                        )
                        col_accum.add_(block_sq.sum(dim=0))

                    col_mean = col_accum.div_(rows)
                    col_sq.mul_(beta2).add_(col_mean, alpha=1.0 - beta2)
                    row_mean_all = row_sq.mean().clamp_min(eps)

                    # Pass 2: precondition gradients in place and measure update RMS.
                    update_sq_sum = torch.zeros((), dtype=torch.float64,
                                                device=parameter.device)
                    for start in range(0, rows, chunk_rows):
                        end = min(start + chunk_rows, rows)
                        variance = (
                            row_sq[start:end, None] * col_sq[None, :]
                            / row_mean_all
                        ).clamp_min(eps)
                        g2[start:end].div_(variance.sqrt())
                        update_sq_sum.add_(
                            g2[start:end].double().square().sum()
                        )

                    update_rms = math.sqrt(
                        float(update_sq_sum.item()) / max(parameter.numel(), 1)
                    )
                    clip_scale = max(1.0, update_rms / clip_threshold)

                    # Pass 3: apply the clipped update without a full-size temp tensor.
                    for start in range(0, rows, chunk_rows):
                        end = min(start + chunk_rows, rows)
                        if weight_decay:
                            p2[start:end].mul_(1.0 - lr * weight_decay)
                        g2[start:end].mul_(1.0 / clip_scale)
                        p2[start:end].add_(g2[start:end], alpha=-lr)

                else:
                    if "sq" not in state:
                        state["sq"] = torch.zeros_like(
                            parameter, dtype=torch.float32, memory_format=torch.preserve_format
                        )
                    sq = state["sq"]
                    if sq.shape != parameter.shape:
                        raise ValueError("optimizer state shape does not match parameter")
                    sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    grad.div_(sq.clamp_min(eps).sqrt())
                    update_rms = float(grad.square().mean().sqrt().item())
                    clip_scale = max(1.0, update_rms / clip_threshold)
                    if weight_decay:
                        parameter.mul_(1.0 - lr * weight_decay)
                    grad.mul_(1.0 / clip_scale)
                    parameter.add_(grad, alpha=-lr)

        return loss


def estimate_state_bytes(parameters: Iterable[torch.nn.Parameter]) -> int:
    """Estimate FP32 factored moment storage without allocating optimizer state."""
    total = 0
    for parameter in parameters:
        if parameter.ndim >= 2:
            rows = parameter.numel() // parameter.shape[-1]
            total += (rows + parameter.shape[-1]) * 4
        else:
            total += parameter.numel() * 4
    return total
