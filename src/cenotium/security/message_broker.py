"""Secure message broker for inter-agent communication."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Set


class MessageType(Enum):
    TRUST_UPDATE = "trust_update"
    AGENT_RESULT = "agent_result"
    SYSTEM_ALERT = "system_alert"
    AGENT_HEARTBEAT = "agent_heartbeat"
    SCHEMA_UPDATE = "schema_update"


@dataclass
class SecureMessage:
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Any
    timestamp: datetime
    signature: str
    encryption_key_id: str
    priority: int = 0
    ttl: int = 600


class MessageBroker:
    """Handles secure message exchange between agents."""

    def __init__(self, security, storage, max_rate: int = 100):
        self.security = security
        self.storage = storage
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)
        self.message_history: Dict[str, List[SecureMessage]] = defaultdict(list)
        self.rate_limits: Dict[str, int] = defaultdict(int)
        self.max_rate = max_rate
        self.message_queue = asyncio.PriorityQueue()

    async def publish(self, topic: str, message: SecureMessage):
        """Publish a message to a topic."""
        if not self._check_rate_limit(message.sender_id):
            raise ValueError("Rate limit exceeded")
        if not self.security.verify_signature(message.payload, message.signature):
            raise ValueError("Invalid message signature")

        encrypted_payload = self.security.encrypt_message(message.payload)
        await self.message_queue.put((message.priority, encrypted_payload, message))
        await self._process_message_queue()

    async def _process_message_queue(self):
        """Process messages in priority order."""
        while not self.message_queue.empty():
            priority, encrypted_payload, message = await self.message_queue.get()
            if self._is_message_expired(message):
                continue
            self.message_history[message.message_type.value].append(message)
            await self._notify_subscribers(message)
            self.message_queue.task_done()

    def subscribe(self, topic: str, callback: Callable[[SecureMessage], None]):
        """Subscribe to a topic."""
        self.subscribers[topic].add(callback)

        def unsubscribe():
            self.subscribers[topic].discard(callback)

        return unsubscribe

    async def _notify_subscribers(self, message: SecureMessage):
        """Notify all subscribers of a message."""
        topic = message.message_type.value
        for subscriber in self.subscribers[topic]:
            try:
                await asyncio.create_task(subscriber(message))
            except Exception as e:
                print(f"Error delivering to subscriber: {e}")

    def _check_rate_limit(self, sender_id: str) -> bool:
        """Check if sender is within rate limit."""
        current_rate = self.rate_limits[sender_id]
        if current_rate >= self.max_rate:
            return False
        self.rate_limits[sender_id] += 1
        return True

    def _is_message_expired(self, message: SecureMessage) -> bool:
        """Check if message has expired based on TTL."""
        age = (datetime.now() - message.timestamp).total_seconds()
        return age > message.ttl
