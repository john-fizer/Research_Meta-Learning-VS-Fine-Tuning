# Version 2.0 - Fine-Tuning Comparison

## What's New

Version 2.0 adds **actual fine-tuning** comparison to answer the research question: **"Can adaptive meta-prompting outperform model fine-tuning?"**

### New Features

#### 1. Fine-Tuning Module (`app/meta_learning/fine_tuning/`)

Complete fine-tuning infrastructure:

- **`trainer.py`** - Full fine-tuning pipeline with HuggingFace Transformers
- **`data_formatter.py`** - Dataset formatting for fine-tuning
- **`model_manager.py`** - Model selection and registry

#### 2. Model Support

Out-of-the-box support for:

**Classification Models:**
- DistilBERT (66M params) - Fast, efficient
- BERT (110M params) - Balanced
- RoBERTa (125M params) - Strong performance

**Generation Models:**
- GPT-2 (117M params)
- GPT-2 Medium (345M params)
- TinyLlama (1.1B params)

#### 3. Experiment Integration

New `FineTuningExperiment` class:
- Runs fine-tuning experiments
- Fair comparison with meta-prompting
- Unified evaluation metrics
- Comprehensive result analysis

### Comparison Framework

```python
from app.meta_learning.experiments import ExperimentRunner, FineTuningExperiment
from app.meta_learning.datasets import CommonsenseQALoader

# Initialize
meta_runner = ExperimentRunner(llm_client=your_client)
ft_experiment = FineTuningExperiment(use_gpu=True)
dataset = CommonsenseQALoader()

# Run full comparison
results = await ft_experiment.run_full_comparison(
    dataset_name="commonsense_qa",
    dataset_loader=lambda n: dataset.get_samples(n),
    initial_prompt="Answer this question...",
    meta_prompting_runner=meta_runner,
    num_iterations=10,
    sample_size=100
)

# Results show which approach wins
if results['meta_prompting_wins']:
    print("Meta-prompting outperforms fine-tuning!")
else:
    print("Fine-tuning outperforms meta-prompting!")
```

### Installation

```bash
# Install all dependencies including fine-tuning packages
pip install -r requirements.txt

# New dependencies in v2.0:
# - transformers==4.36.2
# - datasets==2.16.1
# - accelerate==0.25.0
# - sentencepiece==0.1.99
```

### GPU Support

Fine-tuning benefits significantly from GPU:

```python
# Use GPU if available
ft_experiment = FineTuningExperiment(use_gpu=True)

# Or force CPU
ft_experiment = FineTuningExperiment(use_gpu=False)
```

### Quick Start Example

```python
import asyncio
from app.meta_learning.fine_tuning import (
    FineTuningTrainer,
    FineTuningDataFormatter,
    FineTuningConfig
)
from app.meta_learning.datasets import Sentiment140Loader

async def main():
    # Load dataset
    dataset = Sentiment140Loader()
    samples = dataset.get_samples(500)

    # Format for fine-tuning
    formatter = FineTuningDataFormatter()
    formatted = formatter.format_sentiment140(samples)
    train, eval = formatter.split_train_eval(formatted, eval_ratio=0.2)

    # Configure fine-tuning
    config = FineTuningConfig(
        model_name="distilbert-base-uncased",
        num_labels=2,
        num_epochs=3,
        batch_size=16
    )

    # Train
    trainer = FineTuningTrainer(config)
    results = trainer.train(train, eval, experiment_name="sentiment_ft")

    # Evaluate
    metrics = trainer.evaluate_model(eval)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")

asyncio.run(main())
```

## Architecture Changes

### New Directory Structure

```
app/meta_learning/
├── acla/                   # Adaptive Curriculum Learning (v1.0)
├── clrs/                   # Closed-Loop Reinforcement (v1.0)
├── datasets/               # Dataset loaders (v1.0)
├── fine_tuning/            # NEW: Fine-tuning module (v2.0)
│   ├── trainer.py
│   ├── data_formatter.py
│   └── model_manager.py
├── experiments/            # Updated with fine-tuning support
│   ├── runner.py
│   ├── evaluator.py
│   ├── comparator.py
│   └── fine_tuning_experiment.py  # NEW (v2.0)
└── utils/                  # Visualization and logging (v1.0)
```

## Research Questions

### v2.0 Can Now Answer:

1. **Does meta-prompting outperform fine-tuning?**
   - Fair comparison on same datasets
   - Same evaluation metrics
   - Controlled sample sizes

2. **What are the trade-offs?**
   - Training time comparison
   - Resource requirements
   - Convergence characteristics

3. **When should you use each approach?**
   - Dataset size considerations
   - Task complexity factors
   - Resource constraints

## Performance Notes

### Fine-Tuning:
- **GPU Recommended:** 10-100x faster than CPU
- **Memory:** Requires 2-8GB depending on model size
- **Training Time:** 2-10 minutes for small datasets (CPU), seconds with GPU

### Meta-Prompting:
- **LLM API Required:** Uses Claude/GPT for evaluation
- **No GPU Needed:** Runs on any machine
- **Training Time:** Depends on API latency and iterations

## Migration from v1.0

v1.0 code remains fully compatible. Simply add:

```python
from app.meta_learning.experiments import FineTuningExperiment

# Your existing meta-prompting code works unchanged
# Now you can add fine-tuning comparison
ft_experiment = FineTuningExperiment()
```

## Known Limitations

1. **GPU Memory:** Large models may not fit on consumer GPUs
2. **API Costs:** Meta-prompting experiments use LLM APIs
3. **Dataset Size:** Fine-tuning needs larger datasets for best results

## Future Enhancements

Potential v3.0 features:
- LoRA/QLoRA for efficient fine-tuning
- Multi-task learning support
- Distributed training
- More model architectures
- Advanced prompt optimization techniques

## Changelog

### v2.0 (Current)
- ✅ Added fine-tuning module
- ✅ HuggingFace Transformers integration
- ✅ Model manager and registry
- ✅ Fair comparison framework
- ✅ Multiple model support

### v1.0
- ✅ ACLA (Adaptive Curriculum Learning Agent)
- ✅ CLRS (Closed-Loop Reinforcement System)
- ✅ Dataset loaders (CommonsenseQA, Sentiment140)
- ✅ Experiment framework
- ✅ Visualization and logging utilities
