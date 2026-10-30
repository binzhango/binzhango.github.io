class SpaceSaving:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("positive capacity required")
        self.capacity = capacity
        self.counts = {}  # key -> (estimate, maximum overestimate)

    def add(self, key):
        if key in self.counts:
            count, error = self.counts[key]
            self.counts[key] = (count + 1, error)
        elif len(self.counts) < self.capacity:
            self.counts[key] = (1, 0)
        else:
            victim = min(self.counts, key=lambda x: self.counts[x][0])
            count, _ = self.counts.pop(victim)
            self.counts[key] = (count + 1, count)

    def top(self, k):
        return sorted(self.counts.items(), key=lambda x: -x[1][0])[:max(k, 0)]

s = SpaceSaving(2)
for item in ["a"] * 8 + ["b", "c"]:
    s.add(item)
assert s.top(1)[0][0] == "a"
assert len(s.counts) == 2
assert s.counts["c"] == (2, 1)  # true count is between 1 and 2
