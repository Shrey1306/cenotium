"""Flask server for agent orchestration and monitoring."""

import base64
import json
import logging
import queue
import threading
import time
from datetime import datetime

from flask import Response, jsonify

from .base import create_flask_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("orchestrator.log")],
)
logger = logging.getLogger(__name__)

app = create_flask_app(__name__)

cognitive_stream_queue = queue.Queue()
security_events_queue = queue.Queue()
agent_metrics_queue = queue.Queue()
inter_agent_queue = queue.Queue()

_security_protocol = None


def get_security_protocol():
    global _security_protocol
    if _security_protocol is None:
        from src.cenotium.security import SecurityProtocol

        _security_protocol = SecurityProtocol()
    return _security_protocol


def log_security_event(event: dict):
    """Log a security event to the queue."""
    protocol = get_security_protocol()
    event_with_timestamp = {**event, "timestamp": datetime.now().isoformat()}
    encrypted = protocol.encrypt_message(event_with_timestamp)
    signature = protocol.sign_message(event_with_timestamp)
    security_events_queue.put(
        {
            "encrypted_data": encrypted,
            "signature": signature,
            "timestamp": datetime.now().isoformat(),
        }
    )


def log_agent_metrics(metrics: dict):
    """Log agent metrics to the queue."""
    protocol = get_security_protocol()
    metrics_with_timestamp = {**metrics, "timestamp": datetime.now().isoformat()}
    encrypted = protocol.encrypt_message(metrics_with_timestamp)
    signature = protocol.sign_message(metrics_with_timestamp)
    agent_metrics_queue.put(
        {
            "encrypted_data": encrypted,
            "signature": signature,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/stream/security-events")
def security_events_stream():
    """Stream encrypted security events."""

    def generate():
        while True:
            event = security_events_queue.get()
            if isinstance(event.get("encrypted_data"), bytes):
                event["encrypted_data"] = base64.b64encode(
                    event["encrypted_data"]
                ).decode("utf-8")
            yield f"data: {json.dumps(event)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/stream/agent-metrics")
def agent_metrics_stream():
    """Stream encrypted agent metrics."""

    def generate():
        while True:
            metrics = agent_metrics_queue.get()
            if isinstance(metrics.get("encrypted_data"), bytes):
                metrics["encrypted_data"] = base64.b64encode(
                    metrics["encrypted_data"]
                ).decode("utf-8")
            yield f"data: {json.dumps(metrics)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/decoded/security-events", methods=["GET"])
def fetch_decoded_security_event():
    """Fetch and decode a security event."""
    protocol = get_security_protocol()
    if security_events_queue.empty():
        return jsonify({"message": "No security events available"})

    raw_event = security_events_queue.get()
    if isinstance(raw_event.get("encrypted_data"), bytes):
        raw_event["encrypted_data"] = base64.b64encode(
            raw_event["encrypted_data"]
        ).decode("utf-8")

    raw_encrypted = base64.b64decode(raw_event["encrypted_data"])
    decrypted = protocol.decrypt_message(raw_encrypted)
    signature_valid = protocol.verify_signature(
        decrypted, raw_event.get("signature", "")
    )
    return jsonify({"decrypted": decrypted, "signature_valid": signature_valid})


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}


def simulate_events():
    """Generate sample events for testing."""
    while True:
        log_security_event(
            {
                "event_type": "access_attempt",
                "agent_id": "agent_42",
                "severity": "medium",
                "details": "Access attempt logged",
            }
        )
        log_agent_metrics(
            {
                "agent_id": "agent_42",
                "trust_score": 0.85,
                "performance_metrics": {"response_time": 150, "success_rate": 0.95},
            }
        )
        time.sleep(4)


def main(port: int = 8080, simulate: bool = False):
    if simulate:
        simulator = threading.Thread(target=simulate_events, daemon=True)
        simulator.start()
    logger.info(f"Starting orchestrator on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
