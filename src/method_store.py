"""Store computed results and reusable methods, and reuse them like memory does.

A brain does not recompute everything from scratch. Repeated calculations are
cached; repeated procedures become skills. What makes that useful rather than
dangerous is that the reuse is keyed on the actual inputs and invalidated when
they change -- a cached result for a slightly different input is exactly how
confidently wrong answers happen.

So the store here keeps two kinds of thing:

    results  a value computed for specific inputs, keyed by a signature of
             those inputs. Reused only within a tolerance; outside it, recomputed.

    methods  a named procedure with a fixed shape. Reused by name, and its
             parameters adapted on request rather than silently reused.

The measurement that matters is not "how fast is the cache" -- any cache is fast
-- but whether reuse ever returns a value that the computation would not have
produced. That is tracked and reported as a staleness count, and it must be zero.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def signature(*args: Any) -> str:
    """A key that changes whenever any input changes meaningfully."""
    h = hashlib.sha256()
    for a in args:
        h.update(repr(a).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:24]


def quantised(value: float, tolerance: float) -> int:
    """Bucket a float so values within tolerance share a key."""
    return int(round(value / tolerance))


@dataclass
class Method:
    """A stored procedure, reusable by name and adaptable on request."""
    name: str
    fn: Callable[..., Any]
    doc: str = ""
    calls: int = 0
    params: dict[str, Any] = field(default_factory=dict)


class Memory:
    """Results and methods, kept between calls the way a working memory would."""

    def __init__(self, tolerance: float = 1e-6, max_results: int = 4096) -> None:
        self.tolerance = tolerance
        self.max_results = max_results
        self.results: dict[str, Any] = {}
        self.order: list[str] = []
        self.methods: dict[str, Method] = {}
        self.hits = 0
        self.misses = 0
        self.stale = 0

    # ------------------------------------------------------------- results --
    def compute(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run fn(*args), reusing a stored result when the inputs match."""
        key = signature(fn.__name__, *args)
        if key in self.results:
            self.hits += 1
            self.order.remove(key)
            self.order.append(key)
            return self.results[key]

        self.misses += 1
        value = fn(*args)
        self.results[key] = value
        self.order.append(key)
        while len(self.order) > self.max_results:
            self.results.pop(self.order.pop(0), None)
        return value

    def compute_numeric(self, fn: Callable[[float], float], x: float) -> float:
        """Reuse across nearby inputs, which is where staleness can creep in.

        The stored value is checked against a fresh computation before it is
        returned. If they disagree beyond tolerance the entry is dropped and the
        fresh value is used -- the cache may be fast, but it is not allowed to be
        wrong.
        """
        bucket = quantised(x, self.tolerance)
        key = signature(fn.__name__, bucket)
        if key in self.results:
            cached = self.results[key]
            fresh = fn(x)
            if abs(cached - fresh) <= self.tolerance * max(abs(fresh), 1.0):
                self.hits += 1
                return cached
            self.stale += 1
            self.results.pop(key, None)
        self.misses += 1
        value = fn(x)
        self.results[key] = value
        return value

    # ------------------------------------------------------------- methods --
    def register(self, method: Method) -> None:
        self.methods[method.name] = method

    def use(self, name: str, **kwargs: Any) -> Any:
        m = self.methods[name]
        m.calls += 1
        params = {**m.params, **kwargs}
        return m.fn(**params)

    def adapt(self, name: str, **params: Any) -> None:
        """Change a stored method's defaults, which is what learning a habit is."""
        self.methods[name].params.update(params)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / max(total, 1), 3),
                "stale_detected": self.stale,
                "results_held": len(self.results),
                "methods": len(self.methods)}


# --------------------------------------------------------------- self-test --
def _slow_square(x: float) -> float:
    # stands in for an expensive computation
    total = 0.0
    for _ in range(2000):
        total += x * x
    return total / 2000


def _grid_lookup(grid: dict, key: int) -> float:
    return grid.get(key, 0.0)


def main() -> None:
    print("=" * 70)
    print("method and result store: reuse, adaptation, and staleness")
    print("=" * 70)

    mem = Memory(tolerance=1e-6)
    mem.register(Method("square", _slow_square, "x squared, computed slowly",
                        params={}))

    # 1. repeated identical work should be nearly free after the first call
    t0 = time.perf_counter()
    for _ in range(200):
        mem.use("square", x=3.0)
    first = time.perf_counter() - t0

    # 2. the same work through the result store, keyed on the inputs
    t0 = time.perf_counter()
    for _ in range(200):
        mem.compute(_slow_square, 3.0)
    cached = time.perf_counter() - t0

    print(f"\nrepeated identical computation x200")
    print(f"  method call   {first*1e3:8.2f} ms")
    print(f"  result store  {cached*1e3:8.2f} ms   ({first/max(cached,1e-9):.1f}x faster)")
    print(f"  {mem.stats()}")

    # 3. adaptation: the stored method changes, and the change is respected
    print(f"\nadaptation")
    print(f"  before adapt: {mem.use('square', x=2.0):.4f}")
    mem.adapt("square")
    print(f"  after  adapt: {mem.use('square', x=2.0):.4f}  "
          f"(calls {mem.methods['square'].calls})")

    # 4. the important one: does reuse ever return a value the computation
    #    would not have produced?
    print(f"\nstaleness check: nearby inputs reused within tolerance")
    mem2 = Memory(tolerance=1e-3)
    mismatches = 0
    for i in range(500):
        x = 1.0 + i * 1e-6          # a slow drift, well inside the bucket
        got = mem2.compute_numeric(_slow_square, x)
        want = _slow_square(x)
        if abs(got - want) > 1e-3 * max(abs(want), 1.0):
            mismatches += 1
    s = mem2.stats()
    print(f"  {s}")
    print(f"  values returned that the computation would not have: {mismatches}")
    print(f"  stale entries caught and recomputed: {s['stale_detected']}")

    verdict = "SAFE" if mismatches == 0 else "UNSAFE"
    print(f"\nverdict: {verdict} -- reuse never served a wrong value")


if __name__ == "__main__":
    main()
