"""Round-robin OpenAI-compatible proxy over the local vLLM instances.

Distributes /v1/* requests across the 8 vLLM servers started by
start_vllm.sh. Streams response bodies (SSE-safe). Backends that fail a
request are skipped once before the error propagates.
"""

from __future__ import annotations

import asyncio
import itertools

import uvicorn
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

BACKENDS = [f"http://127.0.0.1:{port}" for port in range(8401, 8409)]
LISTEN_PORT = 8400

app = FastAPI()
_cycle = itertools.cycle(range(len(BACKENDS)))
_client: httpx.AsyncClient | None = None


@app.on_event("startup")
async def _startup():
    global _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))


@app.on_event("shutdown")
async def _shutdown():
    if _client:
        await _client.aclose()


@app.get("/health")
async def health():
    statuses = {}
    for backend in BACKENDS:
        try:
            r = await _client.get(f"{backend}/health")
            statuses[backend] = r.status_code
        except Exception as e:
            statuses[backend] = f"down: {type(e).__name__}"
    return statuses


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "accept-encoding")
    }
    start = next(_cycle)
    last_error: Exception | None = None
    for attempt in range(len(BACKENDS)):
        backend = BACKENDS[(start + attempt) % len(BACKENDS)]
        try:
            req = _client.build_request(
                request.method, f"{backend}/v1/{path}",
                content=body, headers=headers,
            )
            resp = await _client.send(req, stream=True)
            return StreamingResponse(
                _stream(resp),
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type"),
            )
        except Exception as e:  # backend down mid-flight: try next once each
            last_error = e
            continue
    return JSONResponse({"error": f"all backends failed: {last_error}"}, 502)


async def _stream(resp):
    try:
        async for chunk in resp.aiter_raw():
            yield chunk
    finally:
        await resp.aclose()


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    uvicorn.run(app, host="127.0.0.1", port=LISTEN_PORT, log_level="warning")
