"""Trust management using EigenTrust algorithm."""

from datetime import datetime
from typing import Dict, List

import numpy as np


class GlobalTrustCore:
    """Manages global trust scores using modified EigenTrust algorithm."""

    def __init__(
        self,
        storage,
        alpha: float = 0.85,
        trust_threshold: float = 0.5,
        max_iterations: int = 100,
        time_decay_factor: float = 0.95,
    ):
        self.storage = storage
        self.alpha = alpha
        self.trust_threshold = trust_threshold
        self.max_iterations = max_iterations
        self.time_decay_factor = time_decay_factor

    def calculate_trust_score(
        self, agent_id: str, transaction_history: List[dict]
    ) -> float:
        """Calculate trust score using modified EigenTrust with temporal decay."""
        if not transaction_history:
            return 0.0

        weighted_transactions = self._apply_time_decay(transaction_history)

        successful = sum(t["weight"] for t in weighted_transactions if t.get("success"))
        total_weight = sum(t["weight"] for t in weighted_transactions)
        local_trust = successful / total_weight if total_weight > 0 else 0

        neighbor_scores = []
        for transaction in weighted_transactions:
            partner_score = self.storage.get_trust_score(transaction["partner_id"])
            neighbor_scores.append(partner_score * transaction["weight"])

        if not neighbor_scores:
            return local_trust

        global_trust = sum(neighbor_scores) / total_weight
        final_trust = self.alpha * local_trust + (1 - self.alpha) * global_trust
        return max(0.0, min(1.0, final_trust))

    def _apply_time_decay(self, transactions: List[dict]) -> List[dict]:
        """Apply temporal decay to transaction weights."""
        now = datetime.now()
        weighted_transactions = []

        for transaction in transactions:
            age = now - transaction["timestamp"]
            weight = self.time_decay_factor ** (age.days + age.seconds / 86400)
            weighted_transactions.append({**transaction, "weight": weight})

        return weighted_transactions

    def calculate_rank(
        self, agent_id: str, trust_score: float, performance_metrics: dict
    ) -> float:
        """Calculate agent ranking based on multiple factors."""
        weights = {
            "response_time": 0.2,
            "success_rate": 0.3,
            "trust_score": 0.3,
            "complexity_handling": 0.2,
        }

        normalized_metrics = {
            "response_time": 1.0
            / (1.0 + performance_metrics.get("avg_response_time", 1.0)),
            "success_rate": performance_metrics.get("success_rate", 0.0),
            "trust_score": trust_score,
            "complexity_handling": performance_metrics.get("complexity_score", 0.5),
        }

        return sum(weights[k] * normalized_metrics[k] for k in weights)

    def update_trust_network(self, agents: Dict[str, float]) -> Dict[str, float]:
        """Update trust scores for the entire network using power iteration."""
        n = len(agents)
        if n == 0:
            return {}

        trust_matrix = np.zeros((n, n))
        agent_ids = list(agents.keys())

        for i, agent_id in enumerate(agent_ids):
            for j, other_id in enumerate(agent_ids):
                if i != j:
                    trust_matrix[i][j] = self.storage.get_trust_score(other_id)

        row_sums = trust_matrix.sum(axis=1)
        trust_matrix = np.divide(
            trust_matrix,
            row_sums[:, np.newaxis],
            where=row_sums[:, np.newaxis] != 0,
        )

        trust_vector = np.ones(n) / n
        for _ in range(self.max_iterations):
            new_trust = trust_matrix.T @ trust_vector
            if np.allclose(trust_vector, new_trust):
                break
            trust_vector = new_trust

        return {
            agent_id: float(trust_vector[i]) for i, agent_id in enumerate(agent_ids)
        }
