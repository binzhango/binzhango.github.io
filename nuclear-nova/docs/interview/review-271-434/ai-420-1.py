import asyncio
import json

async def stream_tokens(receive, send, factory):
    async def disconnected():
        while True:
            if (await receive())['type'] == 'http.disconnect':
                return

    async def produce():
        stream = factory()
        try:
            await send({'type': 'http.response.start', 'status': 200,
                        'headers': [(b'content-type', b'text/event-stream')]})
            async for token in stream:
                body = ('data: ' + json.dumps({'token': token}) + '\n\n').encode()
                await send({'type': 'http.response.body', 'body': body,
                            'more_body': True})
            await send({'type': 'http.response.body', 'body': b'',
                        'more_body': False})
        finally:
            await stream.aclose()

    tasks = [asyncio.create_task(produce()), asyncio.create_task(disconnected())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task  # propagate backend, send, and receive errors
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

async def test():
    events, closed = asyncio.Queue(), []
    async def receive(): return await events.get()
    async def generator():
        try:
            yield 'first\nline'
            await asyncio.Event().wait()  # simulate a slow next token
        finally: closed.append(True)
    sent = []
    async def send(event):
        sent.append(event)
        if event.get('body'):
            await events.put({'type': 'http.disconnect'})
    await asyncio.wait_for(stream_tokens(receive, send, generator), 1)
    assert closed == [True]
    assert b'first\\nline' in sent[1]['body']

asyncio.run(test())
