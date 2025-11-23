"""
Fine-Tuning Module

Implements model fine-tuning for comparison against meta-prompting.
Supports local fine-tuning of small language models using HuggingFace.
"""

from .trainer import FineTuningTrainer
from .data_formatter import FineTuningDataFormatter
from .model_manager import ModelManager

__all__ = [
    'FineTuningTrainer',
    'FineTuningDataFormatter',
    'ModelManager',
]
