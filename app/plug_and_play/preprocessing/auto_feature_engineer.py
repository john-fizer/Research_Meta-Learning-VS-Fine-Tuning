"""
AutoFeatureEngineer - Intelligent automatic feature engineering.

Advanced feature engineering that goes beyond basic preprocessing:
- Interaction features (multiplicative, additive)
- Polynomial features (2nd, 3rd degree)
- Statistical aggregations (groupby features)
- Domain-specific feature extraction
- Ratio and proportion features
- Binning and discretization
- Feature crosses
- Automated feature selection
- Feature importance ranking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import (
    mutual_info_classif, mutual_info_regression,
    SelectKBest, f_classif, f_regression
)
from itertools import combinations
import warnings

from app.plug_and_play.core.data_analyzer import DataCharacteristics
from app.plug_and_play.core.problem_classifier import ProblemDefinition, ProblemType


class AutoFeatureEngineer:
    """
    Intelligent automated feature engineering.

    Creates new features based on:
    - Data characteristics
    - Problem type
    - Feature relationships
    - Domain patterns
    """

    def __init__(
        self,
        max_interaction_depth: int = 2,
        max_polynomial_degree: int = 2,
        enable_text_features: bool = True,
        enable_time_features: bool = True,
        enable_statistical_features: bool = True,
        feature_selection_k: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize AutoFeatureEngineer.

        Args:
            max_interaction_depth: Max features to combine (default: 2)
            max_polynomial_degree: Max polynomial degree (default: 2)
            enable_text_features: Create advanced text features (default: True)
            enable_time_features: Create advanced time features (default: True)
            enable_statistical_features: Create stat features (default: True)
            feature_selection_k: Keep top K features (None = all)
            verbose: Print progress (default: True)
        """
        self.max_interaction_depth = max_interaction_depth
        self.max_polynomial_degree = max_polynomial_degree
        self.enable_text_features = enable_text_features
        self.enable_time_features = enable_time_features
        self.enable_statistical_features = enable_statistical_features
        self.feature_selection_k = feature_selection_k
        self.verbose = verbose

        self.created_features = []
        self.feature_importance = {}
        self.original_features = []

    def engineer_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        characteristics: DataCharacteristics,
        problem_definition: ProblemDefinition
    ) -> pd.DataFrame:
        """
        Perform comprehensive automated feature engineering.

        Args:
            X: Input features
            y: Target variable (optional)
            characteristics: Data characteristics
            problem_definition: Problem definition

        Returns:
            Enhanced DataFrame with engineered features
        """
        X = X.copy()
        self.original_features = X.columns.tolist()
        initial_count = len(X.columns)

        if self.verbose:
            print("\n" + "="*60)
            print("AUTOMATED FEATURE ENGINEERING")
            print("="*60)
            print(f"\nStarting features: {initial_count}")

        # 1. Numerical interactions
        if len(characteristics.numerical_features) >= 2:
            X = self._create_numerical_interactions(X, characteristics)

        # 2. Polynomial features (for simple problems)
        if (problem_definition.problem_type in [ProblemType.LINEAR_REGRESSION,
                                                 ProblemType.NONLINEAR_REGRESSION] and
            len(characteristics.numerical_features) <= 10):
            X = self._create_polynomial_features(X, characteristics)

        # 3. Ratio and proportion features
        if len(characteristics.numerical_features) >= 2:
            X = self._create_ratio_features(X, characteristics)

        # 4. Statistical aggregations
        if self.enable_statistical_features:
            X = self._create_statistical_features(X, characteristics)

        # 5. Binning/discretization
        X = self._create_binned_features(X, characteristics)

        # 6. Categorical interactions
        if len(characteristics.categorical_features) >= 2:
            X = self._create_categorical_crosses(X, characteristics)

        # 7. Advanced text features
        if self.enable_text_features and characteristics.has_text_component:
            X = self._create_advanced_text_features(X, characteristics)

        # 8. Advanced time features
        if self.enable_time_features and characteristics.has_time_component:
            X = self._create_advanced_time_features(X, characteristics)

        # 9. Domain-specific features
        X = self._create_domain_features(X, characteristics, problem_definition)

        # 10. Feature selection (keep only valuable features)
        if self.feature_selection_k and y is not None:
            X = self._select_best_features(X, y, problem_definition)

        final_count = len(X.columns)

        if self.verbose:
            print(f"\nFinal features: {final_count}")
            print(f"New features created: {final_count - initial_count}")
            print(f"Feature engineering complete!")
            print("="*60 + "\n")

        return X

    def _create_numerical_interactions(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        num_cols = [col for col in c.numerical_features if col in X.columns]

        if len(num_cols) < 2:
            return X

        # Limit to most important combinations
        max_combos = min(20, len(list(combinations(num_cols, 2))))
        created = 0

        for col1, col2 in list(combinations(num_cols, 2))[:max_combos]:
            # Multiplicative interaction
            X[f'{col1}_x_{col2}'] = X[col1] * X[col2]
            created += 1

            # Additive interaction (sum)
            X[f'{col1}_plus_{col2}'] = X[col1] + X[col2]
            created += 1

            # Difference
            X[f'{col1}_minus_{col2}'] = X[col1] - X[col2]
            created += 1

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} numerical interaction features")

        self.created_features.extend([
            f"Numerical interactions: {created} features"
        ])

        return X

    def _create_polynomial_features(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create polynomial features."""
        num_cols = [col for col in c.numerical_features if col in X.columns]

        if len(num_cols) == 0 or len(num_cols) > 10:
            return X  # Skip if too many features

        try:
            poly = PolynomialFeatures(
                degree=self.max_polynomial_degree,
                include_bias=False,
                interaction_only=False
            )

            # Create polynomial features
            X_num = X[num_cols]
            X_poly = poly.fit_transform(X_num)

            # Get feature names
            poly_names = poly.get_feature_names_out(num_cols)

            # Add new polynomial features (exclude original features)
            for i, name in enumerate(poly_names):
                if name not in num_cols:
                    X[f'poly_{name}'] = X_poly[:, i]

            if self.verbose:
                new_poly = len(poly_names) - len(num_cols)
                print(f"  ✓ Created {new_poly} polynomial features")

            self.created_features.append(
                f"Polynomial features (degree {self.max_polynomial_degree})"
            )

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Polynomial features skipped: {e}")

        return X

    def _create_ratio_features(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create ratio and proportion features."""
        num_cols = [col for col in c.numerical_features if col in X.columns]

        if len(num_cols) < 2:
            return X

        created = 0
        max_ratios = min(15, len(list(combinations(num_cols, 2))))

        for col1, col2 in list(combinations(num_cols, 2))[:max_ratios]:
            # Ratio (with safe division)
            denominator = X[col2].replace(0, np.nan)
            X[f'{col1}_div_{col2}'] = X[col1] / denominator
            X[f'{col1}_div_{col2}'].fillna(0, inplace=True)
            created += 1

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} ratio features")

        self.created_features.append(f"Ratio features: {created}")

        return X

    def _create_statistical_features(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create statistical aggregate features."""
        num_cols = [col for col in c.numerical_features if col in X.columns]

        if len(num_cols) < 3:
            return X

        created = 0

        # Row-wise statistics across numerical features
        X['row_mean'] = X[num_cols].mean(axis=1)
        X['row_std'] = X[num_cols].std(axis=1)
        X['row_min'] = X[num_cols].min(axis=1)
        X['row_max'] = X[num_cols].max(axis=1)
        X['row_median'] = X[num_cols].median(axis=1)
        X['row_range'] = X['row_max'] - X['row_min']
        created = 6

        # Coefficient of variation (if mean != 0)
        X['row_cv'] = X['row_std'] / X['row_mean'].replace(0, np.nan)
        X['row_cv'].fillna(0, inplace=True)
        created += 1

        if self.verbose:
            print(f"  ✓ Created {created} statistical features")

        self.created_features.append(f"Statistical features: {created}")

        return X

    def _create_binned_features(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create binned/discretized versions of continuous features."""
        num_cols = [col for col in c.numerical_features if col in X.columns]

        if len(num_cols) == 0:
            return X

        created = 0

        # Bin top numerical features
        for col in num_cols[:5]:  # Limit to top 5
            try:
                # Create quantile-based bins
                X[f'{col}_binned'] = pd.qcut(
                    X[col], q=5, labels=False, duplicates='drop'
                )
                created += 1
            except:
                # Fallback to equal-width bins
                try:
                    X[f'{col}_binned'] = pd.cut(
                        X[col], bins=5, labels=False
                    )
                    created += 1
                except:
                    pass

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} binned features")

        self.created_features.append(f"Binned features: {created}")

        return X

    def _create_categorical_crosses(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create feature crosses between categorical variables."""
        cat_cols = [col for col in c.categorical_features if col in X.columns]

        if len(cat_cols) < 2:
            return X

        created = 0
        max_crosses = min(10, len(list(combinations(cat_cols, 2))))

        for col1, col2 in list(combinations(cat_cols, 2))[:max_crosses]:
            # Check cardinality
            if X[col1].nunique() * X[col2].nunique() > 100:
                continue  # Skip high cardinality crosses

            # Create cross feature
            X[f'{col1}_x_{col2}'] = (
                X[col1].astype(str) + '_' + X[col2].astype(str)
            )
            created += 1

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} categorical cross features")

        self.created_features.append(f"Categorical crosses: {created}")

        return X

    def _create_advanced_text_features(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create advanced text features."""
        text_cols = [col for col in c.text_features if col in X.columns]

        if not text_cols:
            return X

        created = 0

        for col in text_cols[:3]:  # Limit to first 3 text columns
            text_series = X[col].astype(str)

            # Character-level features
            X[f'{col}_num_chars'] = text_series.str.len()
            X[f'{col}_num_words'] = text_series.str.split().str.len()
            X[f'{col}_num_unique_words'] = text_series.apply(
                lambda x: len(set(x.split()))
            )
            X[f'{col}_avg_word_length'] = text_series.apply(
                lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
            )

            # Punctuation and special chars
            X[f'{col}_num_punctuation'] = text_series.str.count(r'[.,!?;:]')
            X[f'{col}_num_capitals'] = text_series.str.count(r'[A-Z]')
            X[f'{col}_num_digits'] = text_series.str.count(r'\d')

            # Lexical diversity (unique words / total words)
            X[f'{col}_lexical_diversity'] = (
                X[f'{col}_num_unique_words'] / X[f'{col}_num_words'].replace(0, 1)
            )

            created += 8

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} advanced text features")

        self.created_features.append(f"Advanced text features: {created}")

        return X

    def _create_advanced_time_features(
        self, X: pd.DataFrame, c: DataCharacteristics
    ) -> pd.DataFrame:
        """Create advanced time-based features."""
        # Time features are already created in AutoPreprocessor
        # This could add more advanced temporal patterns if needed

        created = 0

        # Look for existing time-based columns
        time_cols = [col for col in X.columns if any(
            t in col for t in ['_year', '_month', '_day', '_hour']
        )]

        if time_cols:
            # Is weekend?
            day_cols = [c for c in X.columns if '_dayofweek' in c]
            for col in day_cols:
                X[f'{col}_is_weekend'] = (X[col] >= 5).astype(int)
                created += 1

            # Is business hours? (9-17)
            hour_cols = [c for c in X.columns if '_hour' in c]
            for col in hour_cols:
                X[f'{col}_is_business_hours'] = (
                    (X[col] >= 9) & (X[col] <= 17)
                ).astype(int)
                created += 1

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} advanced time features")

        return X

    def _create_domain_features(
        self, X: pd.DataFrame, c: DataCharacteristics,
        p: ProblemDefinition
    ) -> pd.DataFrame:
        """Create domain-specific features based on column names."""
        created = 0

        # Price/Cost related features
        price_cols = [col for col in X.columns if any(
            term in col.lower() for term in ['price', 'cost', 'amount', 'fee']
        )]

        if len(price_cols) >= 2:
            # Total price
            X['total_price'] = X[price_cols].sum(axis=1)
            created += 1

        # Age-related features
        age_cols = [col for col in X.columns if 'age' in col.lower()]
        for col in age_cols:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Age groups
                X[f'{col}_group'] = pd.cut(
                    X[col],
                    bins=[0, 18, 30, 50, 100],
                    labels=['young', 'adult', 'middle', 'senior']
                )
                created += 1

        # Distance/Location features
        if 'latitude' in X.columns and 'longitude' in X.columns:
            # Distance from origin (0, 0)
            X['distance_from_origin'] = np.sqrt(
                X['latitude']**2 + X['longitude']**2
            )
            created += 1

        if self.verbose and created > 0:
            print(f"  ✓ Created {created} domain-specific features")

        return X

    def _select_best_features(
        self, X: pd.DataFrame, y: pd.Series, p: ProblemDefinition
    ) -> pd.DataFrame:
        """Select top K features based on importance."""
        if len(X.columns) <= self.feature_selection_k:
            return X

        try:
            # Choose score function based on problem type
            if 'CLASSIFICATION' in p.problem_type.name:
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression

            # Select top K features
            selector = SelectKBest(score_func=score_func, k=self.feature_selection_k)

            # Handle non-numeric columns
            X_numeric = X.select_dtypes(include=[np.number])

            if len(X_numeric.columns) > 0:
                X_selected = selector.fit_transform(X_numeric, y)
                selected_features = X_numeric.columns[selector.get_support()].tolist()

                # Store feature importance
                scores = selector.scores_
                self.feature_importance = dict(zip(X_numeric.columns, scores))

                if self.verbose:
                    print(f"  ✓ Selected top {len(selected_features)} features")

                return X[selected_features]

        except Exception as e:
            if self.verbose:
                print(f"  ⚠ Feature selection failed: {e}")

        return X

    def get_feature_report(self) -> Dict[str, Any]:
        """Get detailed report of created features."""
        return {
            'original_features': self.original_features,
            'created_features': self.created_features,
            'feature_importance': self.feature_importance,
            'total_new_features': sum(
                int(f.split(':')[-1].split()[0])
                for f in self.created_features
                if ':' in f
            )
        }

    def get_summary(self) -> str:
        """Get human-readable summary."""
        report = self.get_feature_report()

        summary = f"""
Feature Engineering Summary
{'=' * 50}

Original Features: {len(report['original_features'])}
New Features Created: {report['total_new_features']}

Feature Types Created:
{chr(10).join('  - ' + f for f in report['created_features'])}
"""

        if self.feature_importance:
            top_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            summary += f"\nTop 10 Most Important Features:\n"
            for feat, score in top_features:
                summary += f"  - {feat}: {score:.4f}\n"

        return summary
