# Plug-and-Play ML/DL Examples

This directory contains examples demonstrating the Plug-and-Play ML/DL Framework.

## Quick Start

### Simplest Demo (Start Here!)

```bash
python simple_demo.py
```

This creates sample data and runs the complete analysis pipeline in just 3 lines of code.

### Comprehensive Examples

```bash
python example_basic_usage.py
```

This runs through 5 different examples:
1. Binary Classification (Loan Approval)
2. Regression (House Prices)
3. Text Classification (Sentiment Analysis)
4. Quick Analysis Only
5. Convenience Function Usage

## What You'll See

When you run these examples, the system will:
- ✅ Automatically analyze the dataset
- ✅ Detect the problem type
- ✅ Select optimal models
- ✅ Recommend preprocessing strategies
- ✅ Show detailed analysis reports

## Try With Your Own Data

```python
from app.plug_and_play import PlugAndPlayML

model = PlugAndPlayML()
results = model.run("your_data.csv")
```

That's literally all you need!

## Requirements

Make sure you've installed all dependencies:

```bash
pip install -r ../../requirements.txt
```

## Next Steps

After running these examples:
1. Try with your own CSV files
2. Explore different problem types
3. Check out the comprehensive documentation in `PLUG_AND_PLAY_README.md`
4. Integrate with your existing workflows

Happy AutoML-ing! 🚀
