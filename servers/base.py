"""Base utilities for Flask SSE servers."""

import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Generator

from flask import Flask, Response, stream_with_context
from flask_cors import CORS


def create_flask_app(name: str = __name__, enable_cors: bool = True) -> Flask:
    """Create a Flask app with optional CORS support."""
    app = Flask(name)
    if enable_cors:
        CORS(app)
    return app


def default_serializer(obj: Any) -> Any:
    """Default JSON serializer for objects with model_dump or dict methods."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    return str(obj)


def create_sse_stream(
    generator_fn: Callable[[], Generator],
    serializer: Callable[[Any], Any] = default_serializer,
) -> Generator[str, None, None]:
    """Create an SSE stream from a generator function."""
    for item in generator_fn():
        yield f"data: {json.dumps(item, default=serializer)}\n\n"


def create_async_sse_stream(
    async_generator_fn: Callable[[], AsyncGenerator],
    serializer: Callable[[Any], Any] = default_serializer,
    timeout: int = 120,
) -> Generator[str, None, None]:
    """Create an SSE stream from an async generator with timeout."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        gen = async_generator_fn()
        while True:
            try:
                item = loop.run_until_complete(
                    asyncio.wait_for(gen.__anext__(), timeout=timeout)
                )
                yield f"data: {json.dumps(item, default=serializer)}\n\n"
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'error': 'Timeout reached'})}\n\n"
                break
    finally:
        loop.close()


def sse_response(generator: Generator) -> Response:
    """Create an SSE response from a generator."""
    return Response(stream_with_context(generator), mimetype="text/event-stream")
