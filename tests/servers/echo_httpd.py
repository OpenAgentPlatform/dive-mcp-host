"""Simple HTTP echo server for testing the httpx AsyncClient wrapper."""

import json
from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()


@app.api_route(
    "/echo",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
)
async def echo(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """Echo back request information for verification."""
    body = None
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.json()
        except (ValueError, TypeError):
            body = (await request.body()).decode()

    return JSONResponse(
        {
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "authorization": authorization,
            "body": body,
        }
    )


@app.api_route("/stream", methods=["GET", "POST"])
async def stream(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Stream back request info in chunks for testing streaming."""

    async def generate() -> AsyncGenerator[bytes, None]:
        info = {
            "method": request.method,
            "headers": dict(request.headers),
            "authorization": authorization,
        }
        data = json.dumps(info)
        # Yield in chunks to test streaming
        chunk_size = 10
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size].encode()

    return StreamingResponse(generate(), media_type="application/json")


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Echo HTTP server for testing")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
