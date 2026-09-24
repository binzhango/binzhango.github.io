"""Independently recompute the explicitly hypothetical examples in ai-001..107."""
from math import isclose

kv_small = 2 * 32 * 8 * 128 * 2
assert kv_small == 131_072
assert kv_small * 8192 == 1024**3
kv_large = 2 * 80 * 8 * 128 * 2
assert kv_large == 327_680
assert kv_large * 8192 / 1024**3 == 2.5
assert kv_large * 8192 * 16 / 1024**3 == 40
assert 70e9 * 2 / 1e9 == 140
assert 70e9 * 4 / 8 / 1e9 == 35
assert round(3.35e12 / 35e9) == 96
assert 7e9 * (2 + 2 + 4 + 4 + 4) / 1e9 == 112
assert 16 * (4096 + 4096) == 131_072
assert sum([150, 80, 100, 250, 120]) == 700
assert 10_000 * 500 / 5_000 == 1000
# Exact speculative rejection sampling: accepted mass + residual equals p.
p, q = [0.6, 0.3, 0.1], [0.2, 0.5, 0.3]
accepted = [min(a, b) for a, b in zip(p, q)]
rejection = 1 - sum(accepted)
residual = [max(a - b, 0) for a, b in zip(p, q)]
for target, mass, extra in zip(p, accepted, residual):
    assert isclose(mass + rejection * extra / sum(residual), target)
print('15 numerical checks passed (memory, bandwidth, latency, capacity, rejection sampling).')
