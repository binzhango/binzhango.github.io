from collections import deque

class HeavyHitters:
    def __init__(self, capacity):
        if capacity < 1: raise ValueError("positive capacity")
        self.capacity, self.counters, self.total = capacity, {}, 0

    def add(self, key):
        self.total += 1
        if key in self.counters:
            self.counters[key] += 1
        elif len(self.counters) < self.capacity:
            self.counters[key] = 1
        else:
            for item in list(self.counters):
                self.counters[item] -= 1
                if self.counters[item] == 0: del self.counters[item]

    def candidates(self):
        return sorted(self.counters.items(), key=lambda item: -item[1])

class WindowCounter:
    def __init__(self, seconds):
        if seconds < 1: raise ValueError("positive window")
        self.seconds, self.buckets, self.total, self.last = seconds, deque(), 0, -1

    def count(self, now, increment=0):
        if not isinstance(now, int) or now < self.last or increment < 0:
            raise ValueError("ordered integer time and nonnegative count required")
        self.last = now
        while self.buckets and self.buckets[0][0] <= now - self.seconds:
            self.total -= self.buckets.popleft()[1]
        if increment:
            if self.buckets and self.buckets[-1][0] == now:
                self.buckets[-1][1] += increment
            else: self.buckets.append([now, increment])
            self.total += increment
        return self.total

h = HeavyHitters(2)
for key in 'aaaaabbc': h.add(key)
assert h.candidates()[0][0] == 'a' and len(h.counters) <= 2
w = WindowCounter(3)
assert w.count(10, 2) == 2 and w.count(11, 3) == 5
assert w.count(13) == 3 and w.count(20) == 0
