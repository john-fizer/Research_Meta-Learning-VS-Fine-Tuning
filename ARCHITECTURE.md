# Meta-Learning & Self-Optimizing Systems - Architecture

## Overview

This project implements advanced AI engineering frameworks for researching meta-learning, self-optimizing prompts, and closed-loop reinforcement systems.

**Core Research Question:** Can meta-prompting outperform static fine-tuning?

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESEARCH FRAMEWORK                           │
│  Meta-Learning vs Fine-Tuning Comparative Analysis              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                             │
        ▼                                             ▼
┌───────────────────┐                    ┌───────────────────────┐
│       ACLA        │                    │        CLRS           │
│ (Meta-Prompting)  │                    │ (Reinforcement Loop)  │
└───────────────────┘                    └───────────────────────┘
        │                                             │
        ├─── Prompt Evolution                         ├─── Drift Detection
        ├─── Performance Tracking                     ├─── Alignment Scoring
        └─── Curriculum Learning                      └─── Coherence Analysis
```

## Core Components

### 1. Adaptive Curriculum Learning Agent (ACLA)

**Purpose:** Self-optimizing LLM that rewrites its own training prompts to improve task accuracy

**Location:** `/acla/`

**Key Classes:**
- `AdaptiveCurriculumAgent` - Main orchestrator
- `PromptEvolver` - Handles prompt mutation strategies
- `PerformanceTracker` - Monitors metrics and convergence

**Flow:**
```
Initial Prompt → Evaluate → Analyze Performance →
Evolve Prompt → Re-evaluate → Track Improvement →
Repeat until Convergence
```

**Evolution Strategies:**
1. **Performance-based:** Optimize based on accuracy metrics
2. **Error analysis:** Analyze failure patterns
3. **Ablation testing:** Remove/modify components
4. **Chain-of-thought:** Add reasoning steps
5. **Few-shot:** Optimize example selection

**Data Model:**
```python
CurriculumConfig:
  - initial_prompt: str
  - dataset_name: str
  - max_iterations: int
  - min_performance_threshold: float
  - evolution_strategies: List[str]
  - llm_provider: str
  - model_name: str
```

### 2. Closed-Loop Reinforcement System (CLRS)

**Purpose:** Feedback-driven model adaptation with drift monitoring

**Location:** `/clrs/`

**Key Classes:**
- `ClosedLoopSystem` - Main feedback orchestrator
- `DriftDetector` - Monitors behavioral drift
- `AlignmentScorer` - Measures user preference alignment
- `CoherenceAnalyzer` - Detects emergent patterns

**Flow:**
```
Generate Output → Collect Feedback →
Detect Drift → Score Alignment →
Analyze Coherence → Update Model →
Repeat Cycle
```

**Metrics Tracked:**
- **Drift Score:** Vocabulary, distribution, and length changes
- **Alignment Score:** Preference matching
- **Coherence Score:** Lexical, pattern, and diversity metrics

**Training Cycle:**
```python
TrainingCycle:
  - cycle_id: int
  - outputs: List[str]
  - feedback: List[Dict]
  - drift_score: float
  - alignment_score: float
  - coherence_score: float
  - timestamp: datetime
```

### 3. Dataset Loaders

**Purpose:** Standardized data loading for experiments

**Location:** `/datasets/`

**Supported Datasets:**
- **CommonsenseQA:** Question answering with reasoning
- **Sentiment140:** Sentiment classification
- **Base Interface:** Extensible for custom datasets

**Interface:**
```python
BaseDatasetLoader:
  - load_data() → None
  - get_sample(n: int) → List[Dict]
  - validate_sample(sample: Dict) → bool
  - format_prompt(sample: Dict) → str
```

### 4. Experiment Framework

**Purpose:** Orchestrate and compare experiments

**Location:** `/experiments/`

**Key Classes:**
- `ExperimentRunner` - Run comparative studies
- `ModelEvaluator` - Evaluate prompt performance
- `ExperimentComparator` - Statistical comparison

**Experiment Types:**

**A. Meta-Prompting Experiment:**
```python
run_meta_prompting_experiment(
    config: ExperimentConfig,
    dataset_loader: Callable,
    initial_prompt: str
) → ExperimentResult
```

**B. Static Baseline Experiment:**
```python
run_static_baseline_experiment(
    config: ExperimentConfig,
    dataset_loader: Callable,
    prompt: str
) → ExperimentResult
```

**C. Comparison Study:**
```python
run_comparison_study(
    dataset_name: str,
    dataset_loader: Callable,
    initial_prompt: str,
    num_iterations: int,
    sample_size: int
) → Dict[str, Any]
```

**Results Structure:**
```python
ExperimentResult:
  - config: ExperimentConfig
  - final_performance: Dict[str, float]
  - performance_history: List[Dict[str, float]]
  - best_performance: Dict[str, float]
  - improvement: float
  - convergence_iteration: int
  - timestamp: datetime
  - metadata: Dict[str, Any]
```

### 5. Utilities

**Purpose:** Visualization and logging support

**Location:** `/utils/`

**Components:**
- `MetaLearningVisualizer` - Plot performance curves, convergence analysis
- `ExperimentLogger` - Structured logging with experiment tracking

## Data Flow

### ACLA Experiment Flow

```
1. Load Dataset
   ↓
2. Initialize Agent with Config
   ↓
3. For each iteration:
   ├─ Evaluate current prompt
   ├─ Track performance metrics
   ├─ Analyze errors/patterns
   ├─ Evolve prompt using strategy
   └─ Check convergence
   ↓
4. Save results and best prompt
```

### CLRS Feedback Flow

```
1. Generate outputs from inputs
   ↓
2. Collect user feedback
   ↓
3. Analyze cycle:
   ├─ Detect drift (vocabulary, distribution, length)
   ├─ Score alignment (preference matching)
   └─ Analyze coherence (patterns, diversity)
   ↓
4. Update model weights
   ↓
5. Save checkpoint
   ↓
6. Repeat for next cycle
```

## Storage Structure

```
data/meta_learning/
├── experiments/
│   ├── {dataset}_meta_prompting_{timestamp}.json
│   ├── {dataset}_static_baseline_{timestamp}.json
│   └── comparison_{dataset}_{timestamp}.json
│
├── acla/
│   └── {experiment_name}/
│       ├── checkpoints/
│       │   └── iteration_{n}.json
│       └── prompts/
│           └── evolved_prompts.json
│
└── clrs/
    └── {experiment_name}/
        ├── cycles/
        │   └── cycle_{n}.json
        └── feedback/
            └── feedback_history.json
```

## Key Research Metrics

### Performance Metrics
- **Accuracy:** Correct predictions / Total predictions
- **Precision:** True positives / (True positives + False positives)
- **Recall:** True positives / (True positives + False negatives)
- **F1 Score:** Harmonic mean of precision and recall

### Meta-Learning Metrics
- **Improvement Rate:** (Final - Initial) / Initial
- **Convergence Iteration:** Iteration where performance plateaus
- **Strategy Effectiveness:** Which evolution strategies work best
- **Sample Efficiency:** Performance vs dataset size

### CLRS Metrics
- **Drift Score:** Behavioral stability over cycles
- **Alignment Score:** User preference matching
- **Coherence Score:** Internal consistency
- **Feedback Efficiency:** Performance vs feedback amount

## Integration Points

### LLM Providers
- **Anthropic Claude:** Primary provider (claude-3-5-sonnet)
- **OpenAI:** Alternative provider support
- **Custom Clients:** Extensible interface

### Evaluation Functions
```python
async def evaluate_prompt(
    prompt: str,
    samples: List[Dict],
    llm_client: Any,
    dataset_name: str
) → Dict[str, float]
```

### Dataset Loaders
```python
def dataset_loader(n: Optional[int]) → List[Dict]
```

## Configuration Management

All experiments are configured via dataclasses:

```python
CurriculumConfig(
    initial_prompt="...",
    dataset_name="commonsense_qa",
    max_iterations=10,
    min_performance_threshold=0.8,
    llm_provider="anthropic",
    model_name="claude-3-5-sonnet-20241022"
)

ExperimentConfig(
    name="experiment_name",
    description="...",
    dataset_name="...",
    approach="meta_prompting",  # or "static_baseline"
    num_iterations=10,
    sample_size=100
)
```

## Extension Points

### Adding New Datasets
1. Extend `BaseDatasetLoader` in `/datasets/`
2. Implement `load_data()`, `get_sample()`, `format_prompt()`
3. Add to dataset registry

### Adding Evolution Strategies
1. Add strategy to `PromptEvolver` class
2. Implement evolution logic
3. Add to `CurriculumConfig.evolution_strategies`

### Adding Metrics
1. Extend `ModelEvaluator` in `/experiments/evaluator.py`
2. Add metric calculation
3. Update result structures

## Dependencies

See `requirements.txt` for full list:
- **PyTorch:** Deep learning framework
- **NumPy/Pandas:** Data manipulation
- **scikit-learn:** ML utilities and metrics
- **matplotlib:** Visualization
- **requests:** Dataset downloading
- **tqdm:** Progress bars
- **rich:** Enhanced terminal output

## CI/CD Integration

GitHub Actions workflow: `.github/workflows/run_meta_learning.yml`
- Runs experiments on push
- Generates comparison reports
- Archives results

## Best Practices

1. **Always save checkpoints:** Use `save_path` parameter
2. **Track all experiments:** Log configurations and results
3. **Use async/await:** For concurrent evaluations
4. **Validate datasets:** Check sample format before running
5. **Monitor convergence:** Stop when performance plateaus
6. **Compare fairly:** Same dataset size, iterations, and LLM
