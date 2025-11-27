"""
SmartMatcher - Intelligent dataset-to-model matching algorithm.

This is the brain that decides which models and pipelines are best suited
for a given dataset based on comprehensive analysis.

Matching criteria:
- Problem type and complexity
- Dataset size and characteristics
- Feature types and distributions
- Computational constraints
- Accuracy vs speed tradeoffs
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from app.plug_and_play.core.data_analyzer import DataCharacteristics
from app.plug_and_play.core.problem_classifier import (
    ProblemDefinition, ProblemType, TaskComplexity
)


class ModelCategory(Enum):
    """Categories of ML/DL models."""
    TRADITIONAL_ML = "traditional_ml"
    GRADIENT_BOOSTING = "gradient_boosting"
    DEEP_LEARNING = "deep_learning"
    NLP_TRANSFORMER = "nlp_transformer"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"
    META_LEARNING = "meta_learning"


@dataclass
class ModelRecommendation:
    """Recommendation for a specific model."""

    model_name: str
    model_category: ModelCategory
    framework: str
    priority: int  # 1 (highest) to 5 (lowest)
    confidence: float  # 0-1
    reasons: List[str]
    expected_performance: str  # 'excellent', 'good', 'fair'
    training_time: str  # 'fast', 'medium', 'slow'
    hyperparameters: Dict[str, Any]


@dataclass
class PipelineRecommendation:
    """Complete pipeline recommendation."""

    pipeline_name: str
    models: List[ModelRecommendation]
    preprocessing_strategy: str
    training_strategy: str
    evaluation_strategy: str
    ensemble_strategy: Optional[str]
    meta_learning_integration: Optional[str]
    estimated_total_time: str


class SmartMatcher:
    """
    Intelligent matcher that selects optimal models and pipelines.

    Uses sophisticated heuristics and decision trees to match:
    - Dataset characteristics
    - Problem requirements
    - Performance/speed tradeoffs
    - Available computational resources
    """

    def __init__(self, max_models: int = 5, prefer_speed: bool = False):
        """
        Initialize SmartMatcher.

        Args:
            max_models: Maximum number of models to recommend (default: 5)
            prefer_speed: Prefer faster models over accuracy (default: False)
        """
        self.max_models = max_models
        self.prefer_speed = prefer_speed
        self.recommendations = []

    def match(
        self,
        characteristics: DataCharacteristics,
        problem_definition: ProblemDefinition
    ) -> PipelineRecommendation:
        """
        Match dataset to optimal models and pipeline.

        Args:
            characteristics: Dataset characteristics
            problem_definition: Problem classification

        Returns:
            Complete pipeline recommendation
        """
        # Generate model recommendations
        model_recommendations = self._generate_model_recommendations(
            characteristics, problem_definition
        )

        # Sort by priority and confidence
        model_recommendations = sorted(
            model_recommendations,
            key=lambda x: (x.priority, -x.confidence)
        )[:self.max_models]

        # Determine preprocessing strategy
        preprocessing_strategy = self._determine_preprocessing_strategy(
            characteristics, problem_definition
        )

        # Determine training strategy
        training_strategy = self._determine_training_strategy(
            characteristics, problem_definition
        )

        # Determine evaluation strategy
        evaluation_strategy = self._determine_evaluation_strategy(
            problem_definition
        )

        # Determine if ensemble is beneficial
        ensemble_strategy = self._determine_ensemble_strategy(
            characteristics, problem_definition, model_recommendations
        )

        # Determine meta-learning integration
        meta_strategy = self._determine_meta_learning_integration(
            characteristics, problem_definition
        )

        # Estimate total time
        estimated_time = self._estimate_pipeline_time(
            characteristics, model_recommendations
        )

        # Build pipeline recommendation
        pipeline = PipelineRecommendation(
            pipeline_name=self._generate_pipeline_name(problem_definition),
            models=model_recommendations,
            preprocessing_strategy=preprocessing_strategy,
            training_strategy=training_strategy,
            evaluation_strategy=evaluation_strategy,
            ensemble_strategy=ensemble_strategy,
            meta_learning_integration=meta_strategy,
            estimated_total_time=estimated_time
        )

        self.recommendations = model_recommendations
        return pipeline

    def _generate_model_recommendations(
        self,
        c: DataCharacteristics,
        p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Generate model recommendations based on characteristics."""
        recommendations = []

        # Binary Classification
        if p.problem_type == ProblemType.BINARY_CLASSIFICATION:
            recommendations.extend(self._recommend_binary_classification(c, p))

        # Multiclass Classification
        elif p.problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            recommendations.extend(self._recommend_multiclass_classification(c, p))

        # Regression
        elif 'REGRESSION' in p.problem_type.name:
            recommendations.extend(self._recommend_regression(c, p))

        # Clustering
        elif 'CLUSTERING' in p.problem_type.name:
            recommendations.extend(self._recommend_clustering(c, p))

        # NLP/Text
        elif p.problem_type in [ProblemType.TEXT_CLASSIFICATION,
                                ProblemType.SENTIMENT_ANALYSIS]:
            recommendations.extend(self._recommend_nlp(c, p))

        # Time Series
        elif 'TIME_SERIES' in p.problem_type.name:
            recommendations.extend(self._recommend_time_series(c, p))

        # Default: try various models
        else:
            recommendations.extend(self._recommend_default(c, p))

        return recommendations

    def _recommend_binary_classification(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Recommend models for binary classification."""
        models = []

        # Small dataset (< 1000 samples)
        if c.n_samples < 1000:
            models.append(ModelRecommendation(
                model_name="Logistic Regression",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=1,
                confidence=0.9,
                reasons=["Simple and effective for small datasets",
                        "Fast training", "Interpretable"],
                expected_performance="good",
                training_time="fast",
                hyperparameters={'C': 1.0, 'max_iter': 1000}
            ))

        # Medium dataset (1000-10000)
        elif c.n_samples < 10000:
            models.extend([
                ModelRecommendation(
                    model_name="Random Forest",
                    model_category=ModelCategory.TRADITIONAL_ML,
                    framework="scikit-learn",
                    priority=1,
                    confidence=0.85,
                    reasons=["Handles mixed features well", "Robust to outliers"],
                    expected_performance="good",
                    training_time="medium",
                    hyperparameters={'n_estimators': 100, 'max_depth': 20}
                ),
                ModelRecommendation(
                    model_name="XGBoost",
                    model_category=ModelCategory.GRADIENT_BOOSTING,
                    framework="xgboost",
                    priority=1,
                    confidence=0.90,
                    reasons=["Excellent performance", "Handles missing values"],
                    expected_performance="excellent",
                    training_time="medium",
                    hyperparameters={'n_estimators': 100, 'max_depth': 6,
                                   'learning_rate': 0.1}
                )
            ])

        # Large dataset (> 10000)
        else:
            models.extend([
                ModelRecommendation(
                    model_name="LightGBM",
                    model_category=ModelCategory.GRADIENT_BOOSTING,
                    framework="lightgbm",
                    priority=1,
                    confidence=0.92,
                    reasons=["Very fast on large datasets", "Memory efficient",
                            "Excellent accuracy"],
                    expected_performance="excellent",
                    training_time="fast",
                    hyperparameters={'n_estimators': 100, 'num_leaves': 31,
                                   'learning_rate': 0.1}
                ),
                ModelRecommendation(
                    model_name="Neural Network (MLP)",
                    model_category=ModelCategory.DEEP_LEARNING,
                    framework="pytorch",
                    priority=2,
                    confidence=0.85,
                    reasons=["Can learn complex patterns", "Scalable"],
                    expected_performance="excellent",
                    training_time="medium",
                    hyperparameters={'hidden_layers': [128, 64], 'dropout': 0.2}
                )
            ])

        return models

    def _recommend_multiclass_classification(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Recommend models for multiclass classification."""
        models = []

        # Always try XGBoost for multiclass
        models.append(ModelRecommendation(
            model_name="XGBoost Classifier",
            model_category=ModelCategory.GRADIENT_BOOSTING,
            framework="xgboost",
            priority=1,
            confidence=0.88,
            reasons=["Excellent multiclass performance", "Handles imbalance well"],
            expected_performance="excellent",
            training_time="medium",
            hyperparameters={'n_estimators': 100, 'max_depth': 6}
        ))

        # Random Forest as backup
        models.append(ModelRecommendation(
            model_name="Random Forest Classifier",
            model_category=ModelCategory.TRADITIONAL_ML,
            framework="scikit-learn",
            priority=2,
            confidence=0.82,
            reasons=["Robust and reliable", "Good baseline"],
            expected_performance="good",
            training_time="medium",
            hyperparameters={'n_estimators': 100}
        ))

        # Deep learning for large datasets
        if c.n_samples > 10000:
            models.append(ModelRecommendation(
                model_name="Deep Neural Network",
                model_category=ModelCategory.DEEP_LEARNING,
                framework="pytorch",
                priority=2,
                confidence=0.85,
                reasons=["Can handle complex patterns", "Scalable"],
                expected_performance="excellent",
                training_time="slow",
                hyperparameters={'hidden_layers': [256, 128, 64], 'dropout': 0.3}
            ))

        return models

    def _recommend_regression(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Recommend models for regression."""
        models = []

        # Linear regression for linear relationships
        if c.correlation_strength == 'high':
            models.append(ModelRecommendation(
                model_name="Ridge Regression",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=1,
                confidence=0.88,
                reasons=["Strong linear correlations detected", "Regularized"],
                expected_performance="good",
                training_time="fast",
                hyperparameters={'alpha': 1.0}
            ))

        # Gradient boosting for general regression
        models.extend([
            ModelRecommendation(
                model_name="XGBoost Regressor",
                model_category=ModelCategory.GRADIENT_BOOSTING,
                framework="xgboost",
                priority=1,
                confidence=0.90,
                reasons=["Excellent for non-linear relationships"],
                expected_performance="excellent",
                training_time="medium",
                hyperparameters={'n_estimators': 100, 'max_depth': 6}
            ),
            ModelRecommendation(
                model_name="Random Forest Regressor",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=2,
                confidence=0.85,
                reasons=["Robust to outliers", "Good baseline"],
                expected_performance="good",
                training_time="medium",
                hyperparameters={'n_estimators': 100}
            )
        ])

        return models

    def _recommend_clustering(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Recommend models for clustering."""
        models = [
            ModelRecommendation(
                model_name="K-Means",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=1,
                confidence=0.80,
                reasons=["Fast and simple", "Good starting point"],
                expected_performance="good",
                training_time="fast",
                hyperparameters={'n_clusters': 8}
            ),
            ModelRecommendation(
                model_name="DBSCAN",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=2,
                confidence=0.75,
                reasons=["Finds arbitrary shapes", "Handles noise"],
                expected_performance="good",
                training_time="medium",
                hyperparameters={'eps': 0.5, 'min_samples': 5}
            )
        ]

        return models

    def _recommend_nlp(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Recommend models for NLP tasks."""
        models = []

        # Small dataset - traditional ML with TF-IDF
        if c.n_samples < 5000:
            models.append(ModelRecommendation(
                model_name="TF-IDF + Logistic Regression",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=1,
                confidence=0.82,
                reasons=["Effective for small text datasets", "Fast training"],
                expected_performance="good",
                training_time="fast",
                hyperparameters={'max_features': 5000}
            ))

        # Medium to large - transformers
        else:
            models.extend([
                ModelRecommendation(
                    model_name="DistilBERT",
                    model_category=ModelCategory.NLP_TRANSFORMER,
                    framework="transformers",
                    priority=1,
                    confidence=0.90,
                    reasons=["Fast and accurate", "Pre-trained knowledge"],
                    expected_performance="excellent",
                    training_time="medium",
                    hyperparameters={'max_length': 128, 'batch_size': 16}
                ),
                ModelRecommendation(
                    model_name="LSTM",
                    model_category=ModelCategory.DEEP_LEARNING,
                    framework="pytorch",
                    priority=2,
                    confidence=0.80,
                    reasons=["Good for sequential text", "Less resource intensive"],
                    expected_performance="good",
                    training_time="medium",
                    hyperparameters={'hidden_size': 128, 'num_layers': 2}
                )
            ])

        return models

    def _recommend_time_series(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Recommend models for time series."""
        models = [
            ModelRecommendation(
                model_name="Prophet",
                model_category=ModelCategory.TIME_SERIES,
                framework="prophet",
                priority=1,
                confidence=0.85,
                reasons=["Handles seasonality well", "Robust to missing data"],
                expected_performance="good",
                training_time="fast",
                hyperparameters={}
            ),
            ModelRecommendation(
                model_name="LSTM",
                model_category=ModelCategory.DEEP_LEARNING,
                framework="pytorch",
                priority=1,
                confidence=0.88,
                reasons=["Captures long-term dependencies", "Flexible"],
                expected_performance="excellent",
                training_time="medium",
                hyperparameters={'hidden_size': 64, 'num_layers': 2}
            )
        ]

        return models

    def _recommend_default(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> List[ModelRecommendation]:
        """Default recommendations for unknown problems."""
        return [
            ModelRecommendation(
                model_name="Random Forest",
                model_category=ModelCategory.TRADITIONAL_ML,
                framework="scikit-learn",
                priority=1,
                confidence=0.70,
                reasons=["Versatile and robust", "Good starting point"],
                expected_performance="good",
                training_time="medium",
                hyperparameters={'n_estimators': 100}
            )
        ]

    def _determine_preprocessing_strategy(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> str:
        """Determine preprocessing strategy."""
        strategies = []

        if c.missing_percentage > 10:
            strategies.append("advanced_imputation")
        if c.has_text_component:
            strategies.append("nlp_preprocessing")
        if c.has_time_component:
            strategies.append("time_feature_engineering")
        if c.is_imbalanced:
            strategies.append("resampling")

        return " + ".join(strategies) if strategies else "standard"

    def _determine_training_strategy(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> str:
        """Determine training strategy."""
        if c.n_samples > 100000:
            return "mini_batch_training"
        elif c.n_samples < 1000:
            return "k_fold_cross_validation"
        else:
            return "train_test_split"

    def _determine_evaluation_strategy(self, p: ProblemDefinition) -> str:
        """Determine evaluation strategy."""
        if 'CLASSIFICATION' in p.problem_type.name:
            return "classification_metrics"
        elif 'REGRESSION' in p.problem_type.name:
            return "regression_metrics"
        else:
            return "unsupervised_metrics"

    def _determine_ensemble_strategy(
        self, c: DataCharacteristics, p: ProblemDefinition,
        models: List[ModelRecommendation]
    ) -> Optional[str]:
        """Determine if ensemble is beneficial."""
        if c.complexity_score > 0.6 and len(models) >= 3:
            return "voting_ensemble"
        elif c.n_samples > 10000 and p.complexity == TaskComplexity.VERY_COMPLEX:
            return "stacking_ensemble"
        return None

    def _determine_meta_learning_integration(
        self, c: DataCharacteristics, p: ProblemDefinition
    ) -> Optional[str]:
        """Determine meta-learning integration."""
        if p.requires_nlp and c.n_samples > 5000:
            return "ACLA"  # Adaptive Curriculum Learning
        elif c.complexity_score > 0.7:
            return "CLRS"  # Closed-Loop Reinforcement
        return None

    def _estimate_pipeline_time(
        self, c: DataCharacteristics, models: List[ModelRecommendation]
    ) -> str:
        """Estimate total pipeline execution time."""
        # Simple heuristic based on data size and models
        time_score = 0

        # Data size factor
        if c.n_samples > 100000:
            time_score += 3
        elif c.n_samples > 10000:
            time_score += 2
        else:
            time_score += 1

        # Model complexity factor
        for model in models:
            if model.training_time == "slow":
                time_score += 2
            elif model.training_time == "medium":
                time_score += 1

        if time_score <= 3:
            return "< 5 minutes"
        elif time_score <= 6:
            return "5-15 minutes"
        elif time_score <= 10:
            return "15-60 minutes"
        else:
            return "> 1 hour"

    def _generate_pipeline_name(self, p: ProblemDefinition) -> str:
        """Generate descriptive pipeline name."""
        return f"AutoML_{p.problem_name.replace(' ', '_')}_Pipeline"

    def get_summary(self, pipeline: PipelineRecommendation) -> str:
        """Get human-readable summary of recommendations."""
        models_str = "\n".join([
            f"  {i+1}. {m.model_name} ({m.framework})\n"
            f"     Confidence: {m.confidence:.2%} | Performance: {m.expected_performance} | "
            f"Speed: {m.training_time}\n"
            f"     Reasons: {', '.join(m.reasons[:2])}"
            for i, m in enumerate(pipeline.models)
        ])

        summary = f"""
Smart Matching Results
{'=' * 50}

Pipeline: {pipeline.pipeline_name}
Estimated Time: {pipeline.estimated_total_time}

Recommended Models:
{models_str}

Strategies:
  - Preprocessing: {pipeline.preprocessing_strategy}
  - Training: {pipeline.training_strategy}
  - Evaluation: {pipeline.evaluation_strategy}
  - Ensemble: {pipeline.ensemble_strategy or 'Not needed'}
  - Meta-Learning: {pipeline.meta_learning_integration or 'Not applicable'}

Ready to train!
"""
        return summary
