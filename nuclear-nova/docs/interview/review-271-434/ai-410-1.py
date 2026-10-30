import asyncio

class Runtime:
    def __init__(self, device):
        self.device = device
        self.lock = asyncio.Lock()

    async def run(self, program, inputs, timeout=5.0):
        async with self.lock:
            try:
                async def execute():
                    await self.device.stage(program, inputs)
                    await self.device.start()
                    return await self.device.collect()
                return await asyncio.wait_for(execute(), timeout)
            finally:
                await self.device.abort()  # fence and clear, even after success

class FakeDevice:
    def __init__(self): self.active = False; self.cleared = 0
    async def stage(self, program, inputs):
        if program != 'double-v1': raise ValueError('incompatible program')
        self.inputs = inputs
    async def start(self): self.active = True
    async def collect(self):
        await asyncio.sleep(0)
        return [2*x for x in self.inputs]
    async def abort(self): self.active = False; self.cleared += 1

async def test():
    device = FakeDevice(); runtime = Runtime(device)
    assert await runtime.run('double-v1', [2, 3]) == [4, 6]
    assert not device.active and device.cleared == 1
    try: await runtime.run('wrong', [1])
    except ValueError: pass
    else: raise AssertionError('expected validation failure')
    assert device.cleared == 2

asyncio.run(test())
