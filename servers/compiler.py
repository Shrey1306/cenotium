"""Flask server for LLM Compiler streaming."""

from flask import request

from .base import create_async_sse_stream, create_flask_app, sse_response

app = create_flask_app(__name__)


def get_compiler():
    """Import and create compiler instance."""
    from src.cenotium.compiler import LLMCompiler

    return LLMCompiler()


@app.route("/stream", methods=["GET"])
def stream():
    """Stream compiler output as SSE."""
    query = request.args.get(
        "query",
        "Plan a trip to Cabo for 8 people, under $1500/person, 5 nights, 6 days.",
    )

    compiler = get_compiler()

    async def generate():
        async for step in compiler.astream(query):
            yield step

    return sse_response(create_async_sse_stream(generate))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    main()
