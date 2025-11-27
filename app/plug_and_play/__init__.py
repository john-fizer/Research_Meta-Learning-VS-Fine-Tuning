"""
Plug-and-Play ML/DL Framework

A universal, intelligent machine learning system that automatically:
- Analyzes any dataset
- Detects problem type (classification, regression, NLP, etc.)
- Selects optimal models and pipelines
- Generates appropriate visualizations
- Produces comprehensive reports

Usage:
    from app.plug_and_play import PlugAndPlayML

    # That's it - one line!
    model = PlugAndPlayML()
    results = model.run("your_data.csv")
"""

from app.plug_and_play.core.orchestrator import PlugAndPlayML

__version__ = "1.0.0"
__all__ = ["PlugAndPlayML"]
