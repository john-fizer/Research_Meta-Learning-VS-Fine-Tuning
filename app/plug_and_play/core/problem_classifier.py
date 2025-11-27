"""
ProblemClassifier - Intelligent ML problem type detection.

Automatically classifies the ML task based on data characteristics:
- Classification (binary, multiclass, multilabel)
- Regression (linear, nonlinear, time series)
- Clustering (density-based, hierarchical, partitioning)
- NLP tasks (sentiment, classification, generation, QA)
- Time series (forecasting, anomaly detection)
- Anomaly detection
- Recommendation systems
- Multi-modal tasks
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from app.plug_and_play.core.data_analyzer import DataCharacteristics


class ProblemType(Enum):
    """Enumeration of supported ML problem types."""

    # Classification tasks
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"

    # Regression tasks
    LINEAR_REGRESSION = "linear_regression"
    NONLINEAR_REGRESSION = "nonlinear_regression"
    TIME_SERIES_REGRESSION = "time_series_regression"

    # Clustering tasks
    CLUSTERING = "clustering"
    HIERARCHICAL_CLUSTERING = "hierarchical_clustering"
    DENSITY_CLUSTERING = "density_clustering"

    # NLP tasks
    TEXT_CLASSIFICATION = "text_classification"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NER = "named_entity_recognition"
    TEXT_GENERATION = "text_generation"
    QA = "question_answering"
    SUMMARIZATION = "summarization"

    # Time series tasks
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"

    # Recommendation
    RECOMMENDATION = "recommendation"

    # Multi-modal
    MULTIMODAL = "multimodal"

    # Unknown/Custom
    UNKNOWN = "unknown"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class ProblemDefinition:
    """Complete problem definition and characteristics."""

    # Primary problem type
    problem_type: ProblemType
    problem_name: str
    description: str

    # Secondary/related types (for multi-task scenarios)
    secondary_types: List[ProblemType]

    # Task characteristics
    complexity: TaskComplexity
    requires_nlp: bool
    requires_deep_learning: bool
    requires_ensemble: bool
    requires_feature_engineering: bool

    # Recommended approaches
    recommended_algorithms: List[str]
    recommended_frameworks: List[str]

    # Preprocessing requirements
    preprocessing_needs: List[str]

    # Evaluation metrics
    primary_metrics: List[str]
    secondary_metrics: List[str]

    # Special considerations
    special_notes: List[str]

    # Confidence score
    confidence: float  # 0-1 scale


class ProblemClassifier:
    """
    Intelligent classifier that determines the ML problem type.

    Uses data characteristics to identify the most appropriate ML task
    and recommend suitable algorithms and approaches.
    """

    def __init__(self):
        """Initialize ProblemClassifier."""
        self.problem_definition = None

    def classify(self, characteristics: DataCharacteristics) -> ProblemDefinition:
        """
        Classify the ML problem based on data characteristics.

        Args:
            characteristics: DataCharacteristics from DataAnalyzer

        Returns:
            ProblemDefinition with complete problem classification
        """
        # Determine primary problem type
        problem_type, confidence = self._determine_problem_type(characteristics)

        # Determine complexity
        complexity = self._determine_complexity(characteristics)

        # Check for special requirements
        requirements = self._determine_requirements(characteristics, problem_type)

        # Get recommended algorithms
        algorithms = self._get_recommended_algorithms(problem_type, characteristics)

        # Get recommended frameworks
        frameworks = self._get_recommended_frameworks(problem_type, characteristics)

        # Get preprocessing needs
        preprocessing = self._get_preprocessing_needs(characteristics, problem_type)

        # Get evaluation metrics
        metrics = self._get_evaluation_metrics(problem_type)

        # Detect secondary problem types (multi-task scenarios)
        secondary_types = self._detect_secondary_types(characteristics, problem_type)

        # Generate special notes
        notes = self._generate_special_notes(characteristics, problem_type)

        # Build problem definition
        self.problem_definition = ProblemDefinition(
            problem_type=problem_type,
            problem_name=self._get_problem_name(problem_type),
            description=self._get_problem_description(problem_type),
            secondary_types=secondary_types,
            complexity=complexity,
            requires_nlp=requirements['nlp'],
            requires_deep_learning=requirements['deep_learning'],
            requires_ensemble=requirements['ensemble'],
            requires_feature_engineering=requirements['feature_engineering'],
            recommended_algorithms=algorithms,
            recommended_frameworks=frameworks,
            preprocessing_needs=preprocessing,
            primary_metrics=metrics['primary'],
            secondary_metrics=metrics['secondary'],
            special_notes=notes,
            confidence=confidence
        )

        return self.problem_definition

    def _determine_problem_type(self, c: DataCharacteristics) -> tuple[ProblemType, float]:
        """Determine the primary problem type with confidence score."""

        # Rule-based classification with confidence scoring
        confidence = 1.0

        # Text-heavy datasets -> NLP tasks
        if c.has_text_component and len(c.text_features) >= len(c.numerical_features):
            if c.target_type == 'binary' and c.target_column:
                # Could be sentiment or text classification
                target_name = c.target_column.lower()
                if 'sentiment' in target_name or 'polarity' in target_name:
                    return ProblemType.SENTIMENT_ANALYSIS, 0.9
                else:
                    return ProblemType.TEXT_CLASSIFICATION, 0.85

            elif c.target_type == 'multiclass' and c.target_column:
                return ProblemType.TEXT_CLASSIFICATION, 0.85

            elif c.target_column is None:
                # No target - could be clustering or generation
                return ProblemType.CLUSTERING, 0.6

        # Time series datasets
        if c.has_time_component:
            if c.target_type == 'continuous':
                return ProblemType.TIME_SERIES_FORECASTING, 0.9
            elif c.target_type in ['binary', 'multiclass']:
                return ProblemType.TIME_SERIES_REGRESSION, 0.85
            else:
                return ProblemType.ANOMALY_DETECTION, 0.7

        # No target variable -> Unsupervised learning
        if c.target_column is None:
            if c.has_hierarchical_structure:
                return ProblemType.HIERARCHICAL_CLUSTERING, 0.8
            elif c.sparsity > 0.5:
                return ProblemType.DENSITY_CLUSTERING, 0.75
            else:
                return ProblemType.CLUSTERING, 0.8

        # Classification tasks
        if c.target_type == 'binary':
            return ProblemType.BINARY_CLASSIFICATION, 0.95

        if c.target_type == 'multiclass':
            if c.n_classes > 50:
                # High number of classes might indicate recommendation
                confidence = 0.7
            return ProblemType.MULTICLASS_CLASSIFICATION, confidence

        # Regression tasks
        if c.target_type == 'continuous':
            # Check for linearity
            if c.correlation_strength == 'high':
                return ProblemType.LINEAR_REGRESSION, 0.85
            else:
                return ProblemType.NONLINEAR_REGRESSION, 0.8

        # Multi-modal (text + numerical + categorical)
        if (len(c.text_features) > 0 and
            len(c.numerical_features) > 0 and
            len(c.categorical_features) > 0):
            return ProblemType.MULTIMODAL, 0.75

        # Default: unknown
        return ProblemType.UNKNOWN, 0.3

    def _determine_complexity(self, c: DataCharacteristics) -> TaskComplexity:
        """Determine task complexity level."""
        score = 0

        # Factor 1: Dataset size
        if c.n_samples > 100000:
            score += 2
        elif c.n_samples > 10000:
            score += 1

        # Factor 2: Number of features
        if c.n_features > 100:
            score += 2
        elif c.n_features > 20:
            score += 1

        # Factor 3: Feature types diversity
        type_count = sum([
            len(c.text_features) > 0,
            len(c.numerical_features) > 0,
            len(c.categorical_features) > 0,
            len(c.datetime_features) > 0
        ])
        if type_count >= 3:
            score += 2
        elif type_count >= 2:
            score += 1

        # Factor 4: Data quality issues
        if c.missing_percentage > 30:
            score += 1
        if c.has_outliers:
            score += 1

        # Factor 5: Special features
        if c.has_text_component:
            score += 2
        if c.has_time_component:
            score += 1
        if c.has_hierarchical_structure:
            score += 1

        # Map score to complexity
        if score <= 2:
            return TaskComplexity.SIMPLE
        elif score <= 5:
            return TaskComplexity.MODERATE
        elif score <= 8:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX

    def _determine_requirements(self, c: DataCharacteristics,
                                problem_type: ProblemType) -> Dict[str, bool]:
        """Determine what the task requires."""
        return {
            'nlp': c.has_text_component or 'TEXT' in problem_type.name,
            'deep_learning': (
                c.complexity_score > 0.6 or
                c.n_samples > 10000 or
                c.has_text_component or
                'GENERATION' in problem_type.name
            ),
            'ensemble': c.complexity_score > 0.5 and c.n_samples > 1000,
            'feature_engineering': (
                len(c.categorical_features) > 5 or
                c.has_hierarchical_structure
            )
        }

    def _get_recommended_algorithms(self, problem_type: ProblemType,
                                   c: DataCharacteristics) -> List[str]:
        """Get recommended algorithms for the problem type."""
        algorithms = []

        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            algorithms = [
                'Logistic Regression',
                'Random Forest',
                'XGBoost',
                'LightGBM',
                'Neural Network (MLP)'
            ]
            if c.n_samples > 10000:
                algorithms.append('Deep Neural Network')

        elif problem_type == ProblemType.MULTICLASS_CLASSIFICATION:
            algorithms = [
                'Random Forest',
                'XGBoost',
                'LightGBM',
                'Support Vector Machine',
                'Neural Network (MLP)'
            ]

        elif problem_type in [ProblemType.LINEAR_REGRESSION,
                             ProblemType.NONLINEAR_REGRESSION]:
            algorithms = [
                'Linear Regression',
                'Ridge Regression',
                'Random Forest Regressor',
                'XGBoost Regressor',
                'Neural Network (MLP)'
            ]
            if problem_type == ProblemType.NONLINEAR_REGRESSION:
                algorithms.extend(['SVR', 'Gradient Boosting'])

        elif problem_type == ProblemType.CLUSTERING:
            algorithms = [
                'K-Means',
                'DBSCAN',
                'Hierarchical Clustering',
                'Gaussian Mixture Models'
            ]

        elif problem_type in [ProblemType.TEXT_CLASSIFICATION,
                             ProblemType.SENTIMENT_ANALYSIS]:
            algorithms = [
                'TF-IDF + Logistic Regression',
                'BERT',
                'RoBERTa',
                'DistilBERT',
                'LSTM',
                'CNN for Text'
            ]

        elif problem_type == ProblemType.TIME_SERIES_FORECASTING:
            algorithms = [
                'ARIMA',
                'SARIMA',
                'Prophet',
                'LSTM',
                'GRU',
                'Transformer'
            ]

        elif problem_type == ProblemType.ANOMALY_DETECTION:
            algorithms = [
                'Isolation Forest',
                'One-Class SVM',
                'Autoencoder',
                'LSTM Autoencoder'
            ]

        else:
            algorithms = ['Auto-Selected']

        return algorithms

    def _get_recommended_frameworks(self, problem_type: ProblemType,
                                   c: DataCharacteristics) -> List[str]:
        """Get recommended frameworks for the problem type."""
        frameworks = []

        # Always include scikit-learn for traditional ML
        if 'CLASSIFICATION' in problem_type.name or 'REGRESSION' in problem_type.name:
            frameworks.append('scikit-learn')

        # Gradient boosting libraries
        if c.n_samples > 1000 and not c.has_text_component:
            frameworks.extend(['XGBoost', 'LightGBM', 'CatBoost'])

        # Deep learning frameworks
        if c.complexity_score > 0.6 or c.n_samples > 10000:
            frameworks.extend(['PyTorch', 'TensorFlow/Keras'])

        # NLP frameworks
        if c.has_text_component:
            frameworks.extend(['Hugging Face Transformers', 'spaCy', 'NLTK'])

        # Time series frameworks
        if c.has_time_component:
            frameworks.extend(['statsmodels', 'Prophet'])

        # Meta-learning (our custom systems)
        if c.complexity_score > 0.5:
            frameworks.extend(['ACLA', 'CLRS'])

        # RAG for text generation or QA
        if problem_type in [ProblemType.TEXT_GENERATION, ProblemType.QA]:
            frameworks.append('RAG (LangChain)')

        return list(set(frameworks))  # Remove duplicates

    def _get_preprocessing_needs(self, c: DataCharacteristics,
                                problem_type: ProblemType) -> List[str]:
        """Determine preprocessing requirements."""
        needs = []

        if c.missing_percentage > 5:
            needs.append('Missing value imputation')

        if c.has_outliers:
            needs.append('Outlier detection and handling')

        if len(c.categorical_features) > 0:
            needs.append('Categorical encoding (One-Hot/Label/Target)')

        if len(c.numerical_features) > 0:
            needs.append('Feature scaling/normalization')

        if c.has_text_component:
            needs.extend([
                'Text cleaning and tokenization',
                'Vectorization (TF-IDF/Word2Vec/BERT embeddings)'
            ])

        if c.has_time_component:
            needs.append('Time-based feature engineering')

        if c.is_imbalanced:
            needs.append('Class balancing (SMOTE/undersampling)')

        if c.correlation_strength == 'high':
            needs.append('Feature selection/dimensionality reduction')

        if c.sparsity > 0.5:
            needs.append('Sparse matrix handling')

        return needs

    def _get_evaluation_metrics(self, problem_type: ProblemType) -> Dict[str, List[str]]:
        """Get appropriate evaluation metrics."""
        metrics = {
            'primary': [],
            'secondary': []
        }

        if 'CLASSIFICATION' in problem_type.name:
            metrics['primary'] = ['Accuracy', 'F1-Score', 'ROC-AUC']
            metrics['secondary'] = ['Precision', 'Recall', 'Confusion Matrix']

        elif 'REGRESSION' in problem_type.name:
            metrics['primary'] = ['RMSE', 'MAE', 'R²']
            metrics['secondary'] = ['MAPE', 'Residual plots']

        elif 'CLUSTERING' in problem_type.name:
            metrics['primary'] = ['Silhouette Score', 'Davies-Bouldin Index']
            metrics['secondary'] = ['Calinski-Harabasz Index']

        elif 'TEXT' in problem_type.name or 'NLP' in problem_type.name:
            metrics['primary'] = ['Accuracy', 'F1-Score', 'Perplexity']
            metrics['secondary'] = ['BLEU', 'ROUGE', 'Precision', 'Recall']

        else:
            metrics['primary'] = ['Accuracy', 'Loss']
            metrics['secondary'] = ['Custom metrics']

        return metrics

    def _detect_secondary_types(self, c: DataCharacteristics,
                               primary_type: ProblemType) -> List[ProblemType]:
        """Detect secondary problem types (multi-task scenarios)."""
        secondary = []

        # If has text AND numerical features, could be multimodal
        if c.has_text_component and len(c.numerical_features) > 5:
            if primary_type != ProblemType.MULTIMODAL:
                secondary.append(ProblemType.MULTIMODAL)

        # If has time component, could also be anomaly detection
        if c.has_time_component and primary_type != ProblemType.ANOMALY_DETECTION:
            secondary.append(ProblemType.ANOMALY_DETECTION)

        return secondary

    def _generate_special_notes(self, c: DataCharacteristics,
                               problem_type: ProblemType) -> List[str]:
        """Generate special notes and warnings."""
        notes = []

        if c.n_samples < 1000:
            notes.append('⚠️  Small dataset - consider data augmentation or simpler models')

        if c.missing_percentage > 30:
            notes.append('⚠️  High missing data - imputation strategy is critical')

        if c.is_imbalanced:
            notes.append('⚠️  Imbalanced dataset - use appropriate sampling or weighting')

        if c.complexity_score > 0.8:
            notes.append('💡 Complex dataset - ensemble methods recommended')

        if c.has_text_component and c.n_samples < 5000:
            notes.append('💡 Text data with small sample - consider transfer learning')

        if c.n_features > 100:
            notes.append('💡 High-dimensional data - feature selection recommended')

        if c.has_time_component:
            notes.append('💡 Time series data - ensure proper train/test splitting')

        return notes

    def _get_problem_name(self, problem_type: ProblemType) -> str:
        """Get human-readable problem name."""
        names = {
            ProblemType.BINARY_CLASSIFICATION: "Binary Classification",
            ProblemType.MULTICLASS_CLASSIFICATION: "Multi-Class Classification",
            ProblemType.LINEAR_REGRESSION: "Linear Regression",
            ProblemType.NONLINEAR_REGRESSION: "Non-Linear Regression",
            ProblemType.CLUSTERING: "Clustering",
            ProblemType.TEXT_CLASSIFICATION: "Text Classification",
            ProblemType.SENTIMENT_ANALYSIS: "Sentiment Analysis",
            ProblemType.TIME_SERIES_FORECASTING: "Time Series Forecasting",
            ProblemType.ANOMALY_DETECTION: "Anomaly Detection",
            ProblemType.MULTIMODAL: "Multi-Modal Learning",
        }
        return names.get(problem_type, "Unknown Problem Type")

    def _get_problem_description(self, problem_type: ProblemType) -> str:
        """Get detailed problem description."""
        descriptions = {
            ProblemType.BINARY_CLASSIFICATION: "Predicting one of two possible outcomes",
            ProblemType.MULTICLASS_CLASSIFICATION: "Predicting one of multiple possible classes",
            ProblemType.LINEAR_REGRESSION: "Predicting continuous values with linear relationships",
            ProblemType.NONLINEAR_REGRESSION: "Predicting continuous values with complex relationships",
            ProblemType.CLUSTERING: "Grouping similar data points without labels",
            ProblemType.TEXT_CLASSIFICATION: "Categorizing text documents into classes",
            ProblemType.SENTIMENT_ANALYSIS: "Determining sentiment/emotion in text",
            ProblemType.TIME_SERIES_FORECASTING: "Predicting future values based on historical trends",
            ProblemType.ANOMALY_DETECTION: "Identifying unusual patterns or outliers",
            ProblemType.MULTIMODAL: "Learning from multiple data types simultaneously",
        }
        return descriptions.get(problem_type, "Custom machine learning task")

    def get_summary(self) -> str:
        """Get human-readable summary of problem classification."""
        if self.problem_definition is None:
            return "No classification performed yet. Call classify() first."

        p = self.problem_definition

        secondary_str = ", ".join([s.value for s in p.secondary_types]) if p.secondary_types else "None"

        summary = f"""
Problem Classification Summary
{'=' * 50}

Problem Type: {p.problem_name}
Description: {p.description}
Complexity: {p.complexity.value.upper()}
Confidence: {p.confidence:.2%}

Secondary Types: {secondary_str}

Requirements:
  - NLP Processing: {'Yes' if p.requires_nlp else 'No'}
  - Deep Learning: {'Yes' if p.requires_deep_learning else 'No'}
  - Ensemble Methods: {'Yes' if p.requires_ensemble else 'No'}
  - Feature Engineering: {'Yes' if p.requires_feature_engineering else 'No'}

Recommended Algorithms:
  {chr(10).join('  - ' + algo for algo in p.recommended_algorithms[:5])}

Recommended Frameworks:
  {chr(10).join('  - ' + fw for fw in p.recommended_frameworks[:5])}

Preprocessing Needs:
  {chr(10).join('  - ' + need for need in p.preprocessing_needs[:5])}

Evaluation Metrics:
  Primary: {', '.join(p.primary_metrics)}
  Secondary: {', '.join(p.secondary_metrics)}

Special Notes:
  {chr(10).join('  ' + note for note in p.special_notes)}
"""
        return summary
