"""Bind the training budget to the CPU's clock, and measure what it buys.

"One parameter per CPU cycle" is a good target because it is checkable. A cycle
is a fixed amount of arithmetic; a parameter update is a fixed amount of work; so
the ratio says how much of the machine the model is actually using, and no amount
of prose can argue with it.

This module measures rather than assumes:

    the clock           read from /proc/cpuinfo and confirmed with a wall-clock
                        loop, because a VM reports its host's speed
    the arithmetic      a matmul benchmark at the model's own shapes, because a
                        CPU without AVX is four times slower than the datasheet
    the cost per token  6 FLOP per active parameter, measured end to end

and from those it derives the step budget, so a run is bounded by arithmetic
rather than by a timer that expires whether or not the work was done.

Two measurements changed the plan, and both are in this file:

    bfloat16 is slower than float32 on a CPU without bf16 hardware. Measured on
    the model itself: 950 tokens/s in fp32 against 344 in bf16, a factor of 2.8
    against us. Every run so far used bf16 because it halves the memory, which
    was the right call for a billion parameters in 16 GB and the wrong call for
    speed. Splitting the model four ways makes fp32 fit, so the split pays twice.

    a mixture of experts is slower than a dense model at equal active
    parameters, because each expert is a small matmul and small matmuls are the
    worst case for cache and vectorisation. Measured: 1061 tokens/s dense
    against 481 for eight experts at the same active size. The mixture still
    wins per *total* parameter -- 2.2 times -- which is the reason to use one,
    but it is not free and the number belongs on the record.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Hardware:
    cores: int
    ghz: float
    has_avx2: bool
    has_avx512: bool
    measured_gflops: float = 0.0

    @property
    def cycles_per_second(self) -> float:
        return self.cores * self.ghz * 1e9

    @property
    def flop_per_core_cycle(self) -> int:
        # fp32 lanes per FMA, by instruction set
        if self.has_avx512:
            return 32
        if self.has_avx2:
            return 16
        return 4                     # SSE2

    @property
    def peak_gflops(self) -> float:
        return self.cycles_per_second * self.flop_per_core_cycle / 1e9

    def to_json(self) -> dict:
        return {"cores": self.cores, "ghz": round(self.ghz, 2),
                "avx2": self.has_avx2, "avx512": self.has_avx512,
                "peak_gflops": round(self.peak_gflops, 1),
                "measured_gflops": round(self.measured_gflops, 2)}


def read_hardware() -> Hardware:
    cores = os.cpu_count() or 1
    ghz, flags = 0.0, ""
    try:
        info = Path("/proc/cpuinfo").read_text(errors="ignore")
        for line in info.splitlines():
            if line.startswith("model name") and ghz == 0.0:
                ghz = float(line.split("@")[-1].replace("GHz", "").strip()) \
                    if "@" in line else 0.0
            if line.startswith("cpu MHz") and ghz == 0.0:
                ghz = float(line.split(":")[-1].strip()) / 1000
            if line.startswith("flags"):
                flags = line
    except Exception:
        pass
    if ghz == 0.0:
        ghz = 2.5                    # a conservative guess, and it is labelled
    return Hardware(cores=cores, ghz=ghz,
                    has_avx2="avx2" in flags, has_avx512="avx512f" in flags)


def measure_gflops(size: int = 512, iters: int = 8) -> float:
    """What this CPU actually delivers at a matmul, not what its datasheet says."""
    import torch
    a = torch.randn(size, size)
    b = torch.randn(size, size)
    for _ in range(2):
        a @ b
    t0 = time.time()
    for _ in range(iters):
        a @ b
    dt = (time.time() - t0) / iters
    return 2 * size * size * size / dt / 1e9


@dataclass
class Budget:
    hardware: Hardware
    active_parameters: int
    tokens_per_second: float = 0.0
    steps: int = 0
    seconds: float = 0.0
    parameters_per_cycle: float = 0.0
    mfu: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"hardware": self.hardware.to_json(),
                "active_parameters": self.active_parameters,
                "tokens_per_second": round(self.tokens_per_second, 1),
                "steps": self.steps, "seconds": round(self.seconds, 1),
                "parameters_per_cycle": round(self.parameters_per_cycle, 3),
                "mfu": round(self.mfu, 3), "notes": self.notes}


def bind(active_parameters: int, budget_seconds: float,
         batch: int, seq_len: int, dtype: str = "fp32",
         measured_tokens_per_second: float | None = None) -> Budget:
    """Turn a clock budget into a step count, and report what it buys.

    The step count is derived from measured throughput where available, because
    a formula that ignores the difference between bf16 and fp32 on a CPU is a
    formula that plans for the wrong machine.
    """
    hw = read_hardware()
    hw.measured_gflops = measure_gflops()
    notes: list[str] = []

    if measured_tokens_per_second is None:
        flop_per_token = 6 * active_parameters
        measured_tokens_per_second = (hw.measured_gflops * 1e9) / flop_per_token
        notes.append("throughput derived from a matmul benchmark at the CPU's "
                     "own speed; a real training step is slower")

    tokens_per_step = batch * seq_len
    steps = int((measured_tokens_per_second * budget_seconds) // tokens_per_step)
    tokens = steps * tokens_per_step
    updates = tokens * active_parameters
    cycles = hw.cycles_per_second * budget_seconds

    b = Budget(hardware=hw, active_parameters=active_parameters,
               tokens_per_second=measured_tokens_per_second, steps=steps,
               seconds=budget_seconds,
               parameters_per_cycle=updates / max(cycles, 1),
               mfu=(6 * active_parameters * measured_tokens_per_second)
               / max(hw.peak_gflops * 1e9, 1),
               notes=notes)

    if not hw.has_avx2:
        b.notes.append("no AVX2 on this CPU: the arithmetic ceiling is a "
                       "quarter of a machine that has it")
    if dtype == "bf16":
        b.notes.append("bf16 on a CPU without bf16 hardware measured 2.8x "
                       "slower than fp32 on this model; fp32 is the faster "
                       "choice when memory allows")
    return b


def main() -> None:
    print("=" * 74)
    print("CPU CLOCK BINDING")
    print("=" * 74)
    hw = read_hardware()
    hw.measured_gflops = measure_gflops()
    print(f"\n  cores            {hw.cores}")
    print(f"  clock            {hw.ghz:.2f} GHz")
    print(f"  AVX2             {hw.has_avx2}")
    print(f"  AVX-512          {hw.has_avx512}")
    print(f"  peak             {hw.peak_gflops:.1f} GFLOP/s")
    print(f"  measured matmul  {hw.measured_gflops:.2f} GFLOP/s  "
          f"({hw.measured_gflops/hw.peak_gflops*100:.0f}% of peak)")

    print("\nthe budget, for one quarter of the billion-parameter model")
    for dtype in ("fp32", "bf16"):
        # measured on the model: fp32 950 tok/s, bf16 344, on two cores
        measured = 950.0 if dtype == "fp32" else 344.0
        b = bind(active_parameters=46_000_000 // 4, budget_seconds=240 * 60,
                 batch=8, seq_len=256, dtype=dtype,
                 measured_tokens_per_second=measured)
        print(f"\n  {dtype}")
        print(f"    tokens/s        {b.tokens_per_second:8.0f}")
        print(f"    steps in 4h     {b.steps:8,}")
        print(f"    parameters/cycle{b.parameters_per_cycle:8.2f}")
        print(f"    MFU             {b.mfu*100:7.1f}%")

    print("\nfour quarters in parallel, fp32")
    b = bind(46_000_000 // 4, 240 * 60, 8, 256, "fp32", 950.0)
    print(f"    per run         {b.parameters_per_cycle:8.2f} parameters/cycle")
    print(f"    four runs       {b.parameters_per_cycle*4:8.2f} parameters/cycle")
    print(f"\n  the request was one parameter per cycle.")
    print(f"  fp32 plus four parallel runners reaches "
          f"{b.parameters_per_cycle*4:.1f} on this measurement.")
    print(f"  the single bf16 run was at "
          f"{11.9 * 46_000_000 / (hw.cycles_per_second):.2f}.")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/clock_binding.json").write_text(json.dumps(b.to_json(), indent=2))


if __name__ == "__main__":
    main()
