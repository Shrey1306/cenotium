"""Flask servers for individual agents."""

import json

from flask import request

from .base import create_flask_app, sse_response


def create_perplexity_server():
    """Create Flask app for Perplexity agent."""
    app = create_flask_app(__name__)

    @app.route("/stream", methods=["GET"])
    def stream():
        from src.cenotium.agents.perplexity import perplexity_tool

        query = request.args.get("query", "Default search query")

        def generate():
            result = perplexity_tool.func(query)
            yield f"data: {json.dumps({'result': result})}\n\n"

        return sse_response(generate())

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "agent": "perplexity"}

    return app


def create_twilio_server():
    """Create Flask app for Twilio agent."""
    app = create_flask_app(__name__)

    @app.route("/stream", methods=["GET"])
    def stream():
        from src.cenotium.agents.twilio import twilio_tool

        query = request.args.get(
            "query", '{"to_number": "+14709977644", "message": "Test call"}'
        )

        def generate():
            try:
                data = json.loads(query)
            except json.JSONDecodeError:
                data = {"to_number": "+14709977644", "message": "Test call"}
            result = twilio_tool.run(data)
            yield f"data: {json.dumps({'result': result})}\n\n"

        return sse_response(generate())

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "agent": "twilio"}

    return app


perplexity_app = create_perplexity_server()
twilio_app = create_twilio_server()


def run_perplexity_server(port: int = 7000):
    perplexity_app.run(debug=True, port=port)


def run_twilio_server(port: int = 6000):
    twilio_app.run(debug=True, port=port)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        agent = sys.argv[1]
        if agent == "perplexity":
            run_perplexity_server()
        elif agent == "twilio":
            run_twilio_server()
        else:
            print(f"Unknown agent: {agent}")
    else:
        print("Usage: python -m servers.agents [perplexity|twilio]")
