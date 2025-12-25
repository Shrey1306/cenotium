"""AWS Neptune graph database integration."""

import logging

from flask import Flask, jsonify, request
from flask_cors import CORS
from gremlin_python.driver import client, serializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NeptuneClient:
    """Client for AWS Neptune graph database."""

    def __init__(self, endpoint: str, port: str = "8182"):
        url = f"wss://{endpoint}:{port}/gremlin"
        self.client = client.Client(
            url, "g", message_serializer=serializer.GraphSONSerializersV2d0()
        )

    def run_query(self, query: str):
        """Run a Gremlin query against Neptune."""
        callback = self.client.submitAsync(query)
        if callback.result() is not None:
            return callback.result().all().result()
        return None


def create_neptune_app(endpoint: str) -> Flask:
    """Create Flask app for Neptune queries."""
    app = Flask(__name__)
    CORS(app)

    neptune = NeptuneClient(endpoint)

    @app.route("/query", methods=["POST"])
    def query_endpoint():
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data["prompt"]
        try:
            gremlin_query = (
                f"g.V().has('description', textContains('{prompt}')).valueMap()"
            )
            result = neptune.run_query(gremlin_query)
            return jsonify({"response": result})
        except Exception as e:
            logger.exception("Error processing query")
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}

    return app


def main():
    import os

    endpoint = os.getenv("NEPTUNE_ENDPOINT", "localhost")
    app = create_neptune_app(endpoint)
    app.run(host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
