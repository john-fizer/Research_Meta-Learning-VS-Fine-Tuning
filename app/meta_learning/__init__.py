"""
Meta-Learning & Self-Optimizing Systems

Advanced AI engineering framework for researching meta-learning,
self-optimizing prompts, and closed-loop reinforcement systems.

Research Question: Can meta-prompting outperform static fine-tuning?
"""

from . import acla
from . import clrs
from . import datasets
from . import experiments
from . import utils

__version__ = '0.1.0'

__all__ = [
    'acla',
    'clrs',
    'datasets',
    'experiments',
    'utils',
]
