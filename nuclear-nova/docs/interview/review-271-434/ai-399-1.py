from collections import deque
from threading import Condition, Thread
from time import monotonic

class BatchingLogger:
    def __init__(self, sink, capacity=1000, batch_size=100, interval=1.0):
        if capacity < 1 or not 1 <= batch_size <= 100 or interval <= 0:
            raise ValueError("invalid limits")
        self.sink, self.capacity = sink, capacity
        self.batch_size, self.interval = batch_size, interval
        self.queue, self.cv = deque(), Condition()
        self.closed, self.error, self.failed_batch = False, None, None
        self.worker = Thread(target=self._run, daemon=True)
        self.worker.start()

    def log(self, message):
        with self.cv:
            while len(self.queue) >= self.capacity and not self.closed and not self.error:
                self.cv.wait()
            if self.error: raise RuntimeError("sink failed") from self.error
            if self.closed: raise RuntimeError("logger closed")
            self.queue.append((monotonic(), message))
            self.cv.notify_all()

    def _run(self):
        while True:
            with self.cv:
                while not self.queue and not self.closed:
                    self.cv.wait()
                if not self.queue: return
                remaining = self.queue[0][0] + self.interval - monotonic()
                if not self.closed and len(self.queue) < self.batch_size and remaining > 0:
                    self.cv.wait(remaining)
                    continue
                batch = [self.queue.popleft()[1]
                         for _ in range(min(self.batch_size, len(self.queue)))]
                self.cv.notify_all()
            try:
                self.sink(batch)
            except Exception as exc:
                with self.cv:
                    self.error, self.failed_batch = exc, batch
                    self.cv.notify_all()
                return

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.worker.join()
        if self.error: raise RuntimeError("sink failed") from self.error

batches = []
logger = BatchingLogger(batches.append)
producers = [Thread(target=lambda n=n: [logger.log((n, i)) for i in range(70)])
             for n in range(4)]
for thread in producers: thread.start()
for thread in producers: thread.join()
logger.close()
messages = [message for batch in batches for message in batch]
assert len(messages) == len(set(messages)) == 280
assert all(1 <= len(batch) <= 100 for batch in batches)
