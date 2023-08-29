"""
Custom data types for the deep learning
"""

from dataclasses import dataclass


@dataclass
class TrainerState:
    """
    Soft Actor Critic Trainer State class. Is the common state of all the trainers
    And is passed to the callbacks
    """

    epoch: int = 0
    metrics: dict
    networks: dict
    optimizers: dict

    def log_metrics(self, metrics: dict[str, float]):
        """
        Append the metrics to the metrics dict. If the key is not present, it is created.
        Args:
            metrics (dict): Metrics to log
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def get_last_metrics(self) -> dict[str, float]:
        """
        Returns the last metrics logged
        Returns:
            dict: Last metrics logged
        """
        return {key: value[-1] for key, value in self.metrics.items()}
