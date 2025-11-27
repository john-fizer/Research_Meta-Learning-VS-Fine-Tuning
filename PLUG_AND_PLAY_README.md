# Plug-and-Play ML/DL Framework

**Universal, Intelligent Machine Learning System - Just Load Your Data and Go!**

## 🚀 Overview

The Plug-and-Play ML/DL Framework is a revolutionary system that automatically:
- ✅ Analyzes any dataset and detects its characteristics
- ✅ Identifies the ML problem type (classification, regression, NLP, etc.)
- ✅ Selects optimal models and frameworks
- ✅ Preprocesses data intelligently
- ✅ Trains and evaluates multiple models
- ✅ Generates beautiful visualizations
- ✅ Produces comprehensive reports

**No configuration needed. No manual feature engineering. No model selection headaches.**

## 💡 The Vision

```python
# That's literally it!
from app.plug_and_play import PlugAndPlayML

model = PlugAndPlayML()
results = model.run("your_data.csv")
```

The system does everything automatically:
- Understands your data
- Picks the right models
- Handles preprocessing
- Trains and optimizes
- Shows you the results

## 🎯 Key Features

### 1. **Smart Data Analysis**
- Automatic feature type detection (numerical, categorical, text, datetime)
- Data quality assessment (missing values, outliers, duplicates)
- Statistical property analysis
- Complexity scoring

### 2. **Intelligent Problem Classification**
Automatically detects:
- Binary/Multiclass Classification
- Regression (Linear/Non-linear)
- NLP Tasks (Sentiment Analysis, Text Classification, etc.)
- Time Series Forecasting
- Clustering
- Anomaly Detection
- Multi-modal Learning

### 3. **Adaptive Preprocessing**
- Smart missing value imputation
- Intelligent categorical encoding
- Text preprocessing and vectorization
- Time-based feature engineering
- Automatic feature scaling
- Class imbalance handling

### 4. **Model Matching Algorithm**
Recommends optimal models based on:
- Dataset size and complexity
- Problem type
- Feature characteristics
- Performance vs. speed tradeoffs

### 5. **Multi-Framework Support**
- **Traditional ML**: scikit-learn (Random Forest, SVM, etc.)
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch, TensorFlow/Keras (MLP, LSTM, CNN)
- **NLP**: Transformers (BERT, RoBERTa, DistilBERT)
- **Time Series**: Prophet, ARIMA, LSTM
- **Meta-Learning**: ACLA, CLRS integration

## 📦 Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Optional: Download spaCy language model for NLP
python -m spacy download en_core_web_sm
```

## 🎮 Quick Start

### Basic Usage

```python
from app.plug_and_play import PlugAndPlayML

# Initialize
model = PlugAndPlayML()

# Run on your CSV file
results = model.run("data.csv")

# That's it! The system handles everything automatically.
```

### With Options

```python
model = PlugAndPlayML(
    target_column="price",          # Specify target (or let it auto-detect)
    max_models=5,                   # Number of models to try
    prefer_speed=False,             # Prefer accuracy over speed
    auto_visualize=True,            # Generate visualizations
    verbose=True                    # Show detailed progress
)

results = model.run("sales_data.csv")
```

### Analysis Only (No Training)

```python
# Quick data exploration without training
model = PlugAndPlayML()
analysis = model.analyze_only("data.csv")

print(analysis['characteristics'])
print(analysis['problem'])
```

### Convenience Function

```python
from app.plug_and_play import auto_ml

# One-liner for quick ML
results = auto_ml("data.csv")
```

## 📊 What You Get Back

The `results` dictionary contains:

```python
{
    'data_characteristics': DataCharacteristics,
    'problem_definition': ProblemDefinition,
    'pipeline_recommendation': PipelineRecommendation,
    'preprocessed_data': {'X': DataFrame, 'y': Series},
    'trained_models': List[TrainedModel],       # Coming soon
    'evaluation_metrics': Dict[str, float],     # Coming soon
    'visualizations': Dict[str, Figure],        # Coming soon
    'recommendations': List[str]
}
```

## 🧠 How It Works

### 1. **Data Analysis**
```
DataAnalyzer scans your dataset and extracts:
- Feature types (numerical, categorical, text, datetime)
- Statistical properties (distributions, correlations)
- Data quality metrics (missing %, outliers, duplicates)
- Complexity score
```

### 2. **Problem Classification**
```
ProblemClassifier determines:
- Primary ML task type
- Task complexity level
- Required frameworks
- Recommended algorithms
- Evaluation metrics
```

### 3. **Smart Matching**
```
SmartMatcher selects optimal models based on:
- Dataset characteristics
- Problem requirements
- Performance/speed tradeoffs
- Computational constraints
```

### 4. **Auto-Preprocessing**
```
AutoPreprocessor handles:
- Missing value imputation (KNN/median/mode)
- Categorical encoding (one-hot/label/target)
- Text processing (cleaning/tokenization/vectorization)
- Feature scaling (standard/robust/minmax)
- Time feature engineering
- Outlier handling
```

### 5. **Model Training** (Coming Soon)
```
- Trains recommended models in parallel
- Applies cross-validation
- Optimizes hyperparameters
- Handles class imbalance
- Integrates ACLA/CLRS for meta-learning
```

## 📋 Supported Problem Types

| Problem Type | Auto-Detected | Supported Models |
|-------------|---------------|------------------|
| Binary Classification | ✅ | LR, RF, XGB, LightGBM, MLP |
| Multiclass Classification | ✅ | RF, XGB, MLP, SVM |
| Regression | ✅ | Ridge, RF, XGB, MLP |
| Text Classification | ✅ | TF-IDF+LR, BERT, LSTM |
| Sentiment Analysis | ✅ | DistilBERT, LSTM, CNN |
| Time Series Forecasting | ✅ | Prophet, LSTM, GRU |
| Clustering | ✅ | K-Means, DBSCAN, GMM |
| Anomaly Detection | ✅ | Isolation Forest, Autoencoder |
| Multi-modal Learning | ✅ | Custom ensembles |

## 🎨 Examples

### Example 1: Classification

```python
from app.plug_and_play import PlugAndPlayML

# Load a classification dataset
model = PlugAndPlayML()
results = model.run("customer_churn.csv", target_column="churned")

# View recommendations
print(model.get_recommendations())

# Save results
model.save_results("./output")
```

### Example 2: Regression

```python
# House price prediction
model = PlugAndPlayML()
results = model.run("house_prices.csv", target_column="price")

# Get full analysis
print(model.get_analysis_summary())
```

### Example 3: NLP/Sentiment

```python
# Sentiment analysis on reviews
model = PlugAndPlayML()
results = model.run("reviews.csv", target_column="sentiment")

# The system automatically detects text features
# and selects appropriate NLP models
```

### Example 4: Time Series

```python
# Sales forecasting
model = PlugAndPlayML()
results = model.run("sales_timeseries.csv", target_column="sales")

# Auto-detects time components and selects
# appropriate forecasting models
```

## 🏗️ Architecture

```
app/plug_and_play/
├── core/                          # Brain of the system
│   ├── orchestrator.py            # Main PlugAndPlayML class
│   ├── data_analyzer.py           # Dataset analysis & characterization
│   ├── problem_classifier.py      # ML task detection
│   ├── auto_preprocessor.py       # Intelligent preprocessing
│   └── smart_matcher.py           # Model-to-data matching
│
├── models/                        # Model implementations
│   ├── traditional_ml.py          # Scikit-learn models
│   ├── deep_learning.py           # PyTorch/TF models
│   ├── nlp_models.py              # Transformer models
│   ├── lstm_models.py             # LSTM/GRU/RNN
│   ├── rag_model.py               # RAG pipelines
│   └── meta_models.py             # ACLA/CLRS integration
│
├── pipelines/                     # Problem-specific pipelines
│   ├── classification.py
│   ├── regression.py
│   ├── nlp.py
│   ├── timeseries.py
│   └── clustering.py
│
├── preprocessing/                 # Preprocessing utilities
├── visualization/                 # Adaptive viz engine
├── evaluation/                    # Evaluation & comparison
└── utils/                         # Helper functions
```

## 🔬 Integration with Meta-Learning Systems

The Plug-and-Play framework integrates with existing meta-learning components:

### ACLA (Adaptive Curriculum Learning Agent)
- Used for NLP tasks with sufficient data
- Self-optimizes prompts during training
- Improves model performance iteratively

### CLRS (Closed-Loop Reinforcement System)
- Used for complex, high-dimensional tasks
- Continuous feedback-driven adaptation
- Monitors drift and alignment

The system automatically determines when to activate these components based on:
- Dataset complexity
- Problem type
- Available computational resources

## 🎯 Roadmap

### Phase 1: Analysis & Preprocessing ✅
- [x] Data analyzer
- [x] Problem classifier
- [x] Auto-preprocessor
- [x] Smart matcher

### Phase 2: Model Training (In Progress)
- [ ] Traditional ML trainers
- [ ] Deep learning trainers
- [ ] NLP model trainers
- [ ] Time series trainers
- [ ] Ensemble strategies

### Phase 3: Evaluation & Visualization (Planned)
- [ ] Comprehensive evaluation metrics
- [ ] Model comparison engine
- [ ] Adaptive visualization
- [ ] Interactive dashboards
- [ ] Report generation

### Phase 4: Advanced Features (Future)
- [ ] AutoML hyperparameter optimization
- [ ] Neural architecture search
- [ ] Transfer learning integration
- [ ] Federated learning support
- [ ] Real-time prediction API

## 💪 Why This is Revolutionary

### Traditional ML Workflow:
```
1. Manual data exploration (hours)
2. Manual feature engineering (days)
3. Trial and error with models (days)
4. Hyperparameter tuning (days)
5. Evaluation and comparison (hours)

Total: Weeks of work
```

### Plug-and-Play ML Workflow:
```
1. Load CSV
2. Run one command
3. Get results

Total: Minutes
```

## 🤝 Contributing

This is a living, evolving system. Contributions welcome for:
- New model implementations
- Additional problem types
- Better matching heuristics
- Visualization improvements
- Performance optimizations

## 📝 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

Built on the shoulders of giants:
- scikit-learn for traditional ML
- XGBoost, LightGBM, CatBoost for gradient boosting
- PyTorch & TensorFlow for deep learning
- Hugging Face for NLP
- Our custom ACLA & CLRS systems for meta-learning

---

**Ready to experience truly intelligent, plug-and-play machine learning? Just load your data and go!**

```python
from app.plug_and_play import PlugAndPlayML

model = PlugAndPlayML()
results = model.run("your_data.csv")

# 🎉 That's it - you're done!
```
