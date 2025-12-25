"""Persistent storage for agent data and trust scores."""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import redis


class PersistentStorage:
    """Manages persistent storage of agent data using Redis."""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def store_agent_data(self, agent_id: str, data: dict):
        """Store agent-specific data."""
        key = f"agent:{agent_id}"
        serialized_data = {k: json.dumps(v) for k, v in data.items()}
        self.redis_client.hset(key, mapping=serialized_data)

    def get_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """Retrieve agent data."""
        key = f"agent:{agent_id}"
        data = self.redis_client.hgetall(key)
        return {k.decode(): json.loads(v.decode()) for k, v in data.items()}

    def store_trust_score(self, agent_id: str, trust_score: float):
        """Store agent trust score."""
        if not 0 <= trust_score <= 1:
            raise ValueError("Trust score must be between 0 and 1")
        key = f"trust:{agent_id}"
        self.redis_client.set(key, str(trust_score))

    def get_trust_score(self, agent_id: str) -> float:
        """Retrieve agent trust score."""
        key = f"trust:{agent_id}"
        score = self.redis_client.get(key)
        return float(score) if score else 0.0

    def store_transaction(self, transaction_id: str, data: dict, ttl: int = 600):
        """Store transaction data with TTL."""
        key = f"transaction:{transaction_id}"
        self.redis_client.setex(key, ttl, json.dumps(data))

    def get_transaction(self, transaction_id: str) -> Optional[dict]:
        """Retrieve transaction data."""
        key = f"transaction:{transaction_id}"
        data = self.redis_client.get(key)
        return json.loads(data) if data else None

    def store_agent_metrics(self, agent_id: str, metrics: dict):
        """Store agent performance metrics with time-based scoring."""
        key = f"metrics:{agent_id}"
        self.redis_client.zadd(key, {json.dumps(metrics): datetime.now().timestamp()})
        self.redis_client.zremrangebyscore(
            key, "-inf", datetime.now().timestamp() - 86400
        )
