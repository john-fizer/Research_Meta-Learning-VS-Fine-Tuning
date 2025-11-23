"""
Model Manager

Handles model selection, loading, and management for fine-tuning experiments.
"""

from typing import Dict, Optional, List
from pathlib import Path
import json


class ModelManager:
    """
    Manage fine-tuning models

    Provides:
    - Model selection based on task
    - Model loading and caching
    - Performance tracking across models
    """

    # Recommended models for different tasks
    MODELS = {
        "classification": {
            "small": "distilbert-base-uncased",  # 66M params, fast
            "medium": "bert-base-uncased",       # 110M params, balanced
            "large": "roberta-base",             # 125M params, strong
        },
        "generation": {
            "small": "gpt2",                     # 117M params
            "medium": "gpt2-medium",             # 345M params
            "large": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B params
        }
    }

    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize model manager

        Args:
            save_path: Path to save model registry
        """
        self.save_path = save_path or Path("./data/meta_learning/fine_tuning")
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.save_path / "model_registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load model registry from disk"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {"models": {}}

    def _save_registry(self):
        """Save model registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def get_recommended_model(
        self,
        task_type: str,
        size: str = "small"
    ) -> str:
        """
        Get recommended model for task

        Args:
            task_type: "classification" or "generation"
            size: "small", "medium", or "large"

        Returns:
            Model name
        """
        return self.MODELS.get(task_type, {}).get(size, "distilbert-base-uncased")

    def register_model(
        self,
        experiment_name: str,
        model_path: Path,
        metrics: Dict[str, float],
        config: Dict[str, any]
    ):
        """
        Register a trained model

        Args:
            experiment_name: Experiment name
            model_path: Path to saved model
            metrics: Evaluation metrics
            config: Training configuration
        """
        self.registry["models"][experiment_name] = {
            "model_path": str(model_path),
            "metrics": metrics,
            "config": config,
        }
        self._save_registry()

    def get_best_model(self, metric: str = "accuracy") -> Optional[Dict]:
        """
        Get best model by metric

        Args:
            metric: Metric to compare

        Returns:
            Best model info or None
        """
        models = self.registry.get("models", {})
        if not models:
            return None

        best_name = None
        best_score = float('-inf')

        for name, info in models.items():
            score = info.get("metrics", {}).get(metric, float('-inf'))
            if score > best_score:
                best_score = score
                best_name = name

        if best_name:
            return {
                "name": best_name,
                "score": best_score,
                **self.registry["models"][best_name]
            }

        return None

    def list_models(self) -> List[str]:
        """
        List all registered models

        Returns:
            List of model names
        """
        return list(self.registry.get("models", {}).keys())

    def get_model_info(self, experiment_name: str) -> Optional[Dict]:
        """
        Get info for specific model

        Args:
            experiment_name: Experiment name

        Returns:
            Model info or None
        """
        return self.registry.get("models", {}).get(experiment_name)
