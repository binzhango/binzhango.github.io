from collections import Counter
from heapq import nsmallest

def top_queries(queries, k):
    if k < 0:
        raise ValueError("k must be nonnegative")
    counts = Counter(queries)
    return nsmallest(k, counts.items(), key=lambda item: (-item[1], item[0]))

assert top_queries(["b", "a", "b", "a", "c"], 2) == [("a", 2), ("b", 2)]
assert top_queries([], 3) == []
assert top_queries(["x"], 0) == []
assert top_queries(["x"], 9) == [("x", 1)]
