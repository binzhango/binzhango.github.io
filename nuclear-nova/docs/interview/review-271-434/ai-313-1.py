import math

class VAD:
    def __init__(self, noise_db=-60.0, hangover=8):
        self.noise_db = noise_db
        self.hangover = hangover
        self.remaining = 0

    def step(self, frame):
        if not frame:
            return False
        db = 10 * math.log10(sum(x*x for x in frame) / len(frame) + 1e-12)
        margin = 6 if self.remaining else 10
        speech = db > self.noise_db + margin
        if speech:
            self.remaining = self.hangover
        else:
            self.noise_db = 0.98*self.noise_db + 0.02*db
            self.remaining = max(0, self.remaining - 1)
        return speech or self.remaining > 0

v = VAD(hangover=2)
assert not v.step([0.] * 160)
assert v.step([0.1, -0.1] * 80)
assert v.step([0.] * 160)
assert not v.step([0.] * 160)
assert not v.step([])
