class BlockManager:
    def __init__(self, blocks, block_size):
        if blocks < 1 or block_size < 1:
            raise ValueError("positive sizes required")
        self.size = block_size
        self.free_ids = list(range(blocks))
        self.data, self.refs, self.sequences = {}, {}, {}

    def create(self, name):
        if name in self.sequences:
            raise ValueError("sequence already exists")
        self.sequences[name] = []

    def _allocate(self, values=()):
        if not self.free_ids:
            raise MemoryError("KV pool exhausted")
        block = self.free_ids.pop()
        self.data[block] = list(values)
        self.refs[block] = 1
        return block

    def fork(self, source, target):
        if target in self.sequences:
            raise ValueError("target already exists")
        ids = self.sequences[source].copy()
        for block in ids:
            self.refs[block] += 1
        self.sequences[target] = ids

    def append(self, name, token):
        ids = self.sequences[name]
        if not ids or len(self.data[ids[-1]]) == self.size:
            ids.append(self._allocate())
        elif self.refs[ids[-1]] > 1:
            old = ids[-1]
            new = self._allocate(self.data[old])  # allocate before mutating
            self.refs[old] -= 1
            ids[-1] = new
        self.data[ids[-1]].append(token)

    def values(self, name):
        return [x for block in self.sequences[name] for x in self.data[block]]

    def release(self, name):
        for block in self.sequences.pop(name):
            self.refs[block] -= 1
            if self.refs[block] == 0:
                del self.refs[block], self.data[block]
                self.free_ids.append(block)

m = BlockManager(3, 2)
m.create('a'); m.append('a', 1); m.fork('a', 'b')
m.append('b', 2)
assert m.values('a') == [1] and m.values('b') == [1, 2]
m.append('a', 3); m.append('a', 4)
try:
    m.append('b', 5)
    raise AssertionError("expected exhaustion")
except MemoryError:
    pass
assert m.values('b') == [1, 2]
m.release('a'); m.release('b')
assert len(m.free_ids) == 3 and not m.refs
