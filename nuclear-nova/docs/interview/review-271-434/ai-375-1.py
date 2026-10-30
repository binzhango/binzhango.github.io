from array import array
from threading import Lock

class RingBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("positive capacity required")
        self.data = array('d', [0.0]) * capacity
        self.head = self.size = self.dropped = 0
        self.lock = Lock()

    def append(self, value):
        value = float(value)
        with self.lock:
            pos = (self.head + self.size) % len(self.data)
            self.data[pos] = value
            if self.size == len(self.data):
                self.head = (self.head + 1) % len(self.data)
                self.dropped += 1
            else:
                self.size += 1

    def pop(self):
        with self.lock:
            if not self.size:
                raise IndexError("empty buffer")
            value = self.data[self.head]
            self.head = (self.head + 1) % len(self.data)
            self.size -= 1
            return value

r = RingBuffer(2)
for value in [1, 2, 3]: r.append(value)
assert r.dropped == 1 and r.pop() == 2 and r.pop() == 3
try:
    r.pop(); raise AssertionError("expected empty buffer")
except IndexError:
    pass
