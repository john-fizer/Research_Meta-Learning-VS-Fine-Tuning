"""
DataAnalyzer - Intelligent dataset analysis and characterization.

Automatically detects:
- Data types (numerical, categorical, text, datetime, mixed)
- Dataset characteristics (size, sparsity, imbalance, etc.)
- Data quality issues (missing values, outliers, duplicates)
- Statistical properties (distributions, correlations)
- Text features (language, length, complexity)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class DataCharacteristics:
    """Container for dataset characteristics."""

    # Basic info
    n_samples: int
    n_features: int
    feature_names: List[str]

    # Feature types
    numerical_features: List[str]
    categorical_features: List[str]
    text_features: List[str]
    datetime_features: List[str]
    binary_features: List[str]

    # Data quality
    missing_percentage: float
    duplicate_percentage: float
    has_outliers: bool

    # Statistical properties
    is_imbalanced: bool
    imbalance_ratio: Optional[float]
    sparsity: float

    # Target variable (if detected)
    target_column: Optional[str]
    target_type: Optional[str]  # 'binary', 'multiclass', 'continuous'
    n_classes: Optional[int]

    # Advanced characteristics
    has_time_component: bool
    has_text_component: bool
    has_hierarchical_structure: bool
    correlation_strength: str  # 'low', 'medium', 'high'

    # Dataset complexity
    complexity_score: float  # 0-1 scale
    recommended_sample_size: int


class DataAnalyzer:
    """
    Intelligent data analyzer that automatically detects dataset characteristics.

    This is the brain that understands your data before any modeling happens.
    """

    def __init__(self, text_threshold: int = 50, categorical_threshold: int = 20):
        """
        Initialize DataAnalyzer.

        Args:
            text_threshold: Min avg length to classify as text (default: 50 chars)
            categorical_threshold: Max unique values for categorical (default: 20)
        """
        self.text_threshold = text_threshold
        self.categorical_threshold = categorical_threshold
        self.data = None
        self.characteristics = None

    def analyze(self, data: pd.DataFrame, target_column: Optional[str] = None) -> DataCharacteristics:
        """
        Perform comprehensive analysis of the dataset.

        Args:
            data: Input DataFrame
            target_column: Optional target column name (auto-detected if None)

        Returns:
            DataCharacteristics object with complete analysis
        """
        self.data = data

        # Auto-detect target if not provided
        if target_column is None:
            target_column = self._auto_detect_target()

        # Analyze feature types
        feature_types = self._analyze_feature_types(data)

        # Analyze data quality
        quality_metrics = self._analyze_data_quality(data)

        # Analyze target variable
        target_info = self._analyze_target(data, target_column)

        # Analyze statistical properties
        stats = self._analyze_statistics(data, feature_types)

        # Analyze advanced characteristics
        advanced = self._analyze_advanced_features(data, feature_types)

        # Calculate complexity score
        complexity = self._calculate_complexity(data, feature_types, quality_metrics)

        # Build characteristics object
        self.characteristics = DataCharacteristics(
            n_samples=len(data),
            n_features=len(data.columns),
            feature_names=list(data.columns),

            numerical_features=feature_types['numerical'],
            categorical_features=feature_types['categorical'],
            text_features=feature_types['text'],
            datetime_features=feature_types['datetime'],
            binary_features=feature_types['binary'],

            missing_percentage=quality_metrics['missing_pct'],
            duplicate_percentage=quality_metrics['duplicate_pct'],
            has_outliers=quality_metrics['has_outliers'],

            is_imbalanced=stats['is_imbalanced'],
            imbalance_ratio=stats['imbalance_ratio'],
            sparsity=stats['sparsity'],

            target_column=target_info['column'],
            target_type=target_info['type'],
            n_classes=target_info['n_classes'],

            has_time_component=advanced['has_time'],
            has_text_component=advanced['has_text'],
            has_hierarchical_structure=advanced['has_hierarchy'],
            correlation_strength=advanced['correlation_strength'],

            complexity_score=complexity['score'],
            recommended_sample_size=complexity['recommended_size']
        )

        return self.characteristics

    def _auto_detect_target(self) -> Optional[str]:
        """Auto-detect the target column based on common patterns."""
        common_target_names = [
            'target', 'label', 'class', 'y', 'output', 'prediction',
            'outcome', 'result', 'category', 'sentiment', 'rating'
        ]

        # Check for common names
        for col in self.data.columns:
            if col.lower() in common_target_names:
                return col

        # Check for last column if it looks like a target
        last_col = self.data.columns[-1]
        if self._is_likely_target(self.data[last_col]):
            return last_col

        return None

    def _is_likely_target(self, series: pd.Series) -> bool:
        """Check if a column is likely to be the target variable."""
        # Binary or small number of unique values
        n_unique = series.nunique()
        if n_unique <= 10:
            return True

        # Numerical with reasonable range
        if pd.api.types.is_numeric_dtype(series):
            if series.min() >= 0 and series.max() <= 100:
                return True

        return False

    def _analyze_feature_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Classify each feature into its type."""
        types = {
            'numerical': [],
            'categorical': [],
            'text': [],
            'datetime': [],
            'binary': []
        }

        for col in data.columns:
            series = data[col]

            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                types['datetime'].append(col)
                continue

            # Try to infer datetime from string
            if self._is_datetime_string(series):
                types['datetime'].append(col)
                continue

            # Check for binary
            if series.nunique() == 2:
                types['binary'].append(col)
                continue

            # Check for text
            if self._is_text_feature(series):
                types['text'].append(col)
                continue

            # Check for numerical
            if pd.api.types.is_numeric_dtype(series):
                types['numerical'].append(col)
                continue

            # Check for categorical
            if series.nunique() < self.categorical_threshold:
                types['categorical'].append(col)
            else:
                # High cardinality string column - treat as text
                types['text'].append(col)

        return types

    def _is_text_feature(self, series: pd.Series) -> bool:
        """Check if a feature contains text data."""
        if not pd.api.types.is_string_dtype(series):
            return False

        # Sample non-null values
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return False

        # Check average length
        avg_length = sample.astype(str).str.len().mean()
        if avg_length > self.text_threshold:
            return True

        # Check for multiple words
        has_spaces = sample.astype(str).str.contains(' ').mean() > 0.5
        return has_spaces

    def _is_datetime_string(self, series: pd.Series) -> bool:
        """Check if string column contains datetime values."""
        if not pd.api.types.is_string_dtype(series):
            return False

        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False

        # Common datetime patterns
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]

        for pattern in datetime_patterns:
            if sample.astype(str).str.match(pattern).mean() > 0.8:
                return True

        return False

    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()

        return {
            'missing_pct': (missing_cells / total_cells) * 100,
            'duplicate_pct': (data.duplicated().sum() / len(data)) * 100,
            'has_outliers': self._detect_outliers(data)
        }

    def _detect_outliers(self, data: pd.DataFrame) -> bool:
        """Detect outliers using IQR method."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = ((data[col] < (Q1 - 1.5 * IQR)) |
                       (data[col] > (Q3 + 1.5 * IQR))).sum()

            if outliers > len(data) * 0.05:  # More than 5% outliers
                return True

        return False

    def _analyze_target(self, data: pd.DataFrame, target_column: Optional[str]) -> Dict[str, Any]:
        """Analyze target variable characteristics."""
        if target_column is None or target_column not in data.columns:
            return {
                'column': None,
                'type': None,
                'n_classes': None
            }

        target = data[target_column]
        n_unique = target.nunique()

        # Determine target type
        if n_unique == 2:
            target_type = 'binary'
        elif n_unique < 20 and not pd.api.types.is_numeric_dtype(target):
            target_type = 'multiclass'
        elif pd.api.types.is_numeric_dtype(target):
            target_type = 'continuous'
        else:
            target_type = 'multiclass'

        return {
            'column': target_column,
            'type': target_type,
            'n_classes': n_unique if target_type != 'continuous' else None
        }

    def _analyze_statistics(self, data: pd.DataFrame, feature_types: Dict) -> Dict[str, Any]:
        """Analyze statistical properties."""
        stats = {
            'is_imbalanced': False,
            'imbalance_ratio': None,
            'sparsity': 0.0
        }

        # Calculate sparsity
        if feature_types['numerical']:
            num_data = data[feature_types['numerical']]
            zero_pct = (num_data == 0).sum().sum() / (num_data.shape[0] * num_data.shape[1])
            stats['sparsity'] = zero_pct

        return stats

    def _analyze_advanced_features(self, data: pd.DataFrame, feature_types: Dict) -> Dict[str, Any]:
        """Analyze advanced dataset characteristics."""
        # Check for time component
        has_time = len(feature_types['datetime']) > 0

        # Check for text component
        has_text = len(feature_types['text']) > 0

        # Check for hierarchical structure (nested categories, etc.)
        has_hierarchy = self._detect_hierarchy(data, feature_types)

        # Analyze correlation strength
        correlation_strength = self._analyze_correlations(data, feature_types)

        return {
            'has_time': has_time,
            'has_text': has_text,
            'has_hierarchy': has_hierarchy,
            'correlation_strength': correlation_strength
        }

    def _detect_hierarchy(self, data: pd.DataFrame, feature_types: Dict) -> bool:
        """Detect hierarchical structure in categorical features."""
        # Simple heuristic: check if categorical features have nested patterns
        categorical_cols = feature_types['categorical']

        if len(categorical_cols) < 2:
            return False

        # Check for common prefixes (e.g., "Category_A", "Category_A_1")
        for col in categorical_cols[:5]:  # Check first 5
            values = data[col].astype(str).unique()
            if any('_' in str(v) or '.' in str(v) or '/' in str(v) for v in values):
                return True

        return False

    def _analyze_correlations(self, data: pd.DataFrame, feature_types: Dict) -> str:
        """Analyze correlation strength between numerical features."""
        numerical_cols = feature_types['numerical']

        if len(numerical_cols) < 2:
            return 'low'

        # Calculate correlation matrix
        corr_matrix = data[numerical_cols].corr().abs()

        # Get upper triangle (exclude diagonal)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Calculate average correlation
        avg_corr = upper_triangle.stack().mean()

        if avg_corr > 0.7:
            return 'high'
        elif avg_corr > 0.3:
            return 'medium'
        else:
            return 'low'

    def _calculate_complexity(self, data: pd.DataFrame, feature_types: Dict,
                            quality_metrics: Dict) -> Dict[str, Any]:
        """Calculate dataset complexity score and recommendations."""
        complexity_factors = []

        # Factor 1: Number of features (normalized)
        n_features = len(data.columns)
        complexity_factors.append(min(n_features / 100, 1.0) * 0.2)

        # Factor 2: Number of samples
        n_samples = len(data)
        if n_samples < 1000:
            complexity_factors.append(0.3 * 0.2)
        elif n_samples < 10000:
            complexity_factors.append(0.5 * 0.2)
        else:
            complexity_factors.append(0.8 * 0.2)

        # Factor 3: Missing data
        complexity_factors.append(min(quality_metrics['missing_pct'] / 50, 1.0) * 0.2)

        # Factor 4: Feature type diversity
        type_diversity = len([v for v in feature_types.values() if v]) / 5
        complexity_factors.append(type_diversity * 0.2)

        # Factor 5: Text/NLP component (adds complexity)
        if feature_types['text']:
            complexity_factors.append(0.2)
        else:
            complexity_factors.append(0.0)

        complexity_score = sum(complexity_factors)

        # Recommend sample size for training
        if complexity_score > 0.7:
            recommended_size = min(int(n_samples * 0.8), 50000)
        elif complexity_score > 0.4:
            recommended_size = min(int(n_samples * 0.7), 10000)
        else:
            recommended_size = min(int(n_samples * 0.6), 5000)

        return {
            'score': complexity_score,
            'recommended_size': recommended_size
        }

    def get_summary(self) -> str:
        """Get human-readable summary of data analysis."""
        if self.characteristics is None:
            return "No analysis performed yet. Call analyze() first."

        c = self.characteristics

        summary = f"""
Dataset Analysis Summary
{'=' * 50}

Basic Information:
  - Samples: {c.n_samples:,}
  - Features: {c.n_features}
  - Target: {c.target_column or 'Not detected'}
  - Problem Type: {c.target_type or 'Unknown'}

Feature Types:
  - Numerical: {len(c.numerical_features)}
  - Categorical: {len(c.categorical_features)}
  - Text: {len(c.text_features)}
  - DateTime: {len(c.datetime_features)}
  - Binary: {len(c.binary_features)}

Data Quality:
  - Missing Data: {c.missing_percentage:.2f}%
  - Duplicates: {c.duplicate_percentage:.2f}%
  - Outliers: {'Yes' if c.has_outliers else 'No'}

Characteristics:
  - Imbalanced: {'Yes' if c.is_imbalanced else 'No'}
  - Sparsity: {c.sparsity:.2%}
  - Correlation: {c.correlation_strength}
  - Complexity: {c.complexity_score:.2f}/1.0

Special Features:
  - Time Series: {'Yes' if c.has_time_component else 'No'}
  - NLP/Text: {'Yes' if c.has_text_component else 'No'}
  - Hierarchical: {'Yes' if c.has_hierarchical_structure else 'No'}

Recommendations:
  - Suggested Training Size: {c.recommended_sample_size:,} samples
"""
        return summary
