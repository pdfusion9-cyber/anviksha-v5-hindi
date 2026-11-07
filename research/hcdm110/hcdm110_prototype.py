"""HCDM-110 isolated research prototype.

This module is intentionally NOT imported by src/moe_model.py or stream_train.py.
It uses synthetic pooled representations and is only for interface/unit tests.
No pretrained checkpoint is loaded, transformed or written.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
import torch.nn.functional as F


class DGPatternEncoder(nn.Module):
    """Fixed random expansion + ReLU + exact top-k sparsification.

    Input:  [B, D] floating point vectors.
    Output: [B, E] L2-normalised sparse codes with at most K nonzeros.
    The projection is a frozen registered buffer, not a trainable parameter.
    """
    def __init__(self, input_dim: int = 384, expansion_dim: int = 4096,
                 active_k: int = 64, seed: int = 110) -> None:
        super().__init__()
        if min(input_dim, expansion_dim, active_k) <= 0 or active_k > expansion_dim:
            raise ValueError("invalid DG dimensions or active_k")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        projection = torch.randn(input_dim, expansion_dim, generator=gen)
        projection = F.normalize(projection, dim=0)
        self.register_buffer("projection", projection, persistent=True)
        self.active_k = int(active_k)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.projection.shape[0]:
            raise ValueError("DG input must have shape [B, input_dim]")
        if not torch.isfinite(x).all():
            raise ValueError("DG input contains non-finite values")
        scores = F.relu(x.float() @ self.projection.float())
        values, indices = scores.topk(self.active_k, dim=-1)
        sparse = torch.zeros_like(scores)
        sparse.scatter_(1, indices, values)
        return F.normalize(sparse, dim=-1)


class AssociativeMemory:
    """Bounded CPU key-value memory with cosine top-k retrieval.

    Keys are sparse DG codes; values are pooled backbone representations.
    Retrieval is exact cosine search in this prototype (ANN is a later scale step).
    """
    def __init__(self, capacity: int, key_dim: int, value_dim: int) -> None:
        if min(capacity, key_dim, value_dim) <= 0:
            raise ValueError("memory dimensions must be positive")
        self.capacity, self.key_dim, self.value_dim = capacity, key_dim, value_dim
        self.keys = torch.zeros(capacity, key_dim, dtype=torch.float32)
        self.values = torch.zeros(capacity, value_dim, dtype=torch.float32)
        self.valid = torch.zeros(capacity, dtype=torch.bool)
        self.write_pos = 0
        self.size = 0

    @torch.no_grad()
    def add(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.ndim != 2 or keys.shape[1] != self.key_dim:
            raise ValueError("keys must have shape [N, key_dim]")
        if values.ndim != 2 or values.shape != (keys.shape[0], self.value_dim):
            raise ValueError("values must have shape [N, value_dim]")
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError("memory input contains non-finite values")
        count = int(keys.shape[0])
        if count == 0:
            return
        k = F.normalize(keys.detach().float().cpu(), dim=-1)
        v = values.detach().float().cpu()

        # Keep only the newest capacity entries if a single insertion is larger
        # than the ring. Compute their correct final slots without repeated-index
        # advanced assignment.
        if count > self.capacity:
            skipped = count - self.capacity
            k = k[-self.capacity:]
            v = v[-self.capacity:]
            first_slot = (self.write_pos + skipped) % self.capacity
            slots = (torch.arange(self.capacity) + first_slot) % self.capacity
        else:
            slots = (torch.arange(count) + self.write_pos) % self.capacity

        self.keys.index_copy_(0, slots, k)
        self.values.index_copy_(0, slots, v)
        self.valid[slots] = True
        self.write_pos = (self.write_pos + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    @torch.no_grad()
    def retrieve(self, queries: torch.Tensor, top_k: int = 4):
        if queries.ndim != 2 or queries.shape[1] != self.key_dim:
            raise ValueError("queries must have shape [B, key_dim]")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        batch = queries.shape[0]
        if self.size == 0:
            return (torch.zeros(batch, 0, self.value_dim, device=queries.device),
                    torch.zeros(batch, 0, device=queries.device),
                    torch.zeros(batch, 0, dtype=torch.long, device=queries.device))
        slots = torch.nonzero(self.valid, as_tuple=False).flatten()
        q = F.normalize(queries.detach().float().cpu(), dim=-1)
        scores = q @ self.keys[slots].T
        k = min(top_k, slots.numel())
        values, indices = scores.topk(k, dim=-1)
        selected_slots = slots[indices]
        retrieved = self.values[selected_slots]
        return (retrieved.to(queries.device), values.to(queries.device),
                selected_slots.to(queries.device))


class CA1Integrator(nn.Module):
    """Small gated memory integration block with a zero-initialised residual.

    Disagreement is a vector-space distance proxy, NOT a semantic contradiction
    detector. A contradiction detector requires task labels or model predictions.
    """
    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, current: torch.Tensor, retrieved: torch.Tensor):
        if current.ndim != 2 or retrieved.ndim != 3:
            raise ValueError("CA1 expects current [B,D] and retrieved [B,K,D]")
        if current.shape[0] != retrieved.shape[0] or current.shape[-1] != retrieved.shape[-1]:
            raise ValueError("CA1 batch or feature dimensions do not match")
        if retrieved.shape[1] == 0:
            return current, retrieved.new_zeros(current.shape[0], 0), current.new_zeros(current.shape[0])
        q = self.query(current).unsqueeze(1)
        k = self.key(retrieved)
        v = self.value(retrieved)
        weights = torch.softmax((q * k).sum(-1) / (current.shape[-1] ** 0.5), dim=-1)
        context = (weights.unsqueeze(-1) * v).sum(1)
        integrated = current + self.output(context)
        disagreement = 1.0 - F.cosine_similarity(current.float(), context.float(), dim=-1)
        return integrated, weights, disagreement.clamp(0.0, 2.0)


@dataclass
class WorkingMemoryState:
    hidden: torch.Tensor


class PFCWorkingMemory(nn.Module):
    """Document-scoped recurrent objective/context state with three controls."""
    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.cell = nn.GRUCell(dim, dim)
        self.policy = nn.Linear(dim * 2, 3)

    def initial_state(self, batch: int, device=None, dtype=torch.float32) -> WorkingMemoryState:
        return WorkingMemoryState(torch.zeros(batch, self.cell.hidden_size, device=device, dtype=dtype))

    def forward(self, pooled_input: torch.Tensor, state: WorkingMemoryState):
        if pooled_input.ndim != 2 or state.hidden.shape != pooled_input.shape:
            raise ValueError("PFC expects pooled input and state with identical [B,D] shape")
        new_hidden = self.cell(pooled_input, state.hidden)
        controls = torch.softmax(self.policy(torch.cat([pooled_input, new_hidden], dim=-1)), dim=-1)
        return WorkingMemoryState(new_hidden), controls


class ReplayPriority:
    """Quality-gated priority score for deterministic memory sampling."""
    WEIGHTS = {
        "novelty": 0.25,
        "prediction_error": 0.25,
        "quality": 0.30,
        "learning_progress": 0.10,
        "diversity": 0.10,
    }

    @classmethod
    def score(cls, signals: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = set(cls.WEIGHTS) - set(signals)
        if missing:
            raise ValueError(f"missing replay signals: {sorted(missing)}")
        values = [signals[k].float() for k in cls.WEIGHTS]
        if any(v.ndim != 1 for v in values):
            raise ValueError("replay signals must be one-dimensional")
        if len({v.shape for v in values}) != 1:
            raise ValueError("replay signals must have matching shapes")
        out = torch.zeros_like(values[0])
        for key, weight in cls.WEIGHTS.items():
            value = signals[key].float()
            if not torch.isfinite(value).all() or torch.any((value < 0) | (value > 1)):
                raise ValueError(f"{key} must be finite and in [0,1]")
            out.add_(value, alpha=weight)
        return out

    @classmethod
    def select(cls, signals: Mapping[str, torch.Tensor], count: int) -> torch.Tensor:
        scores = cls.score(signals)
        if count <= 0:
            raise ValueError("count must be positive")
        return scores.topk(min(count, scores.numel())).indices


class LowRankAdapter(nn.Module):
    """LoRA-style residual adapter. Zero-init up projection preserves identity."""
    def __init__(self, dim: int = 384, rank: int = 8) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x))


class DualProjectionAdapterBank(nn.Module):
    """Two rank-r adapters per layer (Q and V); 98,304 params at 8x384x8."""
    def __init__(self, layers: int = 8, dim: int = 384, rank: int = 8) -> None:
        super().__init__()
        if layers <= 0:
            raise ValueError("layers must be positive")
        self.q_adapters = nn.ModuleList([LowRankAdapter(dim, rank) for _ in range(layers)])
        self.v_adapters = nn.ModuleList([LowRankAdapter(dim, rank) for _ in range(layers)])

    def forward(self, layer: int, q: torch.Tensor, v: torch.Tensor):
        return self.q_adapters[layer](q), self.v_adapters[layer](v)


@dataclass(frozen=True)
class ControllerSignals:
    novelty: float
    prediction_error: float
    data_quality: float
    retrieval_confidence: float
    learning_progress: float
    expert_utilisation: float
    gradient_norm: float


@dataclass(frozen=True)
class ControllerDecision:
    learning_rate_multiplier: float
    replay_fraction: float
    load_balance_weight_multiplier: float


class AdaptiveLearningController:
    """Bounded rule-based controller; deliberately does not force router choices."""
    @staticmethod
    def decide(s: ControllerSignals) -> ControllerDecision:
        vals = tuple(float(v) for v in s.__dict__.values())
        if any(not torch.isfinite(torch.tensor(v)) for v in vals):
            raise ValueError("controller signals must be finite")
        unit = (s.novelty, s.prediction_error, s.data_quality,
                s.retrieval_confidence, s.learning_progress, s.expert_utilisation)
        if any(v < 0 or v > 1 for v in unit) or s.gradient_norm < 0:
            raise ValueError("controller signals out of range")
        grad_penalty = min(max(s.gradient_norm - 1.0, 0.0), 2.0)
        lr = 0.8 + 0.2*s.novelty + 0.15*s.prediction_error + 0.1*s.learning_progress
        lr -= 0.2*(1.0-s.data_quality) + 0.15*grad_penalty
        replay = 0.05 + 0.25*s.novelty + 0.20*s.prediction_error
        replay += 0.15*(1.0-s.data_quality) - 0.15*s.retrieval_confidence
        balance = 1.0 + 0.5*(1.0-s.expert_utilisation)
        return ControllerDecision(
            learning_rate_multiplier=min(1.25, max(0.5, lr)),
            replay_fraction=min(0.5, max(0.05, replay)),
            load_balance_weight_multiplier=min(2.0, max(0.5, balance)),
        )


class HCDM110Prototype(nn.Module):
    """Isolated subsystem composition for contract tests; not a model wrapper."""
    def __init__(self, dim: int = 384, expanded_dim: int = 4096,
                 dg_active_k: int = 64, layers: int = 8, rank: int = 8) -> None:
        super().__init__()
        self.dg = DGPatternEncoder(dim, expanded_dim, dg_active_k)
        self.ca1 = CA1Integrator(dim)
        self.pfc = PFCWorkingMemory(dim)
        self.adapters = DualProjectionAdapterBank(layers, dim, rank)

    def integrate(self, pooled: torch.Tensor, retrieved: torch.Tensor):
        return self.ca1(pooled, retrieved)

    def parameter_report(self) -> dict[str, int]:
        return {
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
            "adapter_parameters": sum(p.numel() for p in self.adapters.parameters()),
            "dg_frozen_buffer_elements": self.dg.projection.numel(),
        }
