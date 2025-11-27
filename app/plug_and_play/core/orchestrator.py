"""
PlugAndPlayML - Main orchestrator for the intelligent ML/DL system.

This is the user-facing interface that coordinates all components:
- Data analysis
- Problem classification
- Preprocessing
- Model matching and training
- Visualization
- Results reporting

Usage:
    model = PlugAndPlayML()
    results = model.run("data.csv")
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, Any, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from app.plug_and_play.core.data_analyzer import DataAnalyzer, DataCharacteristics
from app.plug_and_play.core.problem_classifier import ProblemClassifier, ProblemDefinition
from app.plug_and_play.core.auto_preprocessor import AutoPreprocessor
from app.plug_and_play.core.smart_matcher import SmartMatcher, PipelineRecommendation
from app.plug_and_play.utils.enhanced_loader import EnhancedDataLoader


class PlugAndPlayML:
    """
    Universal Plug-and-Play ML/DL System.

    Just load your CSV and let the magic happen!

    Features:
    - Automatic data analysis and characterization
    - Intelligent problem type detection
    - Smart preprocessing pipeline
    - Optimal model selection
    - Automated training and evaluation
    - Beautiful visualizations
    - Comprehensive reports
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        max_models: int = 5,
        prefer_speed: bool = False,
        auto_visualize: bool = True,
        verbose: bool = True,
        use_enhanced_loader: bool = True
    ):
        """
        Initialize PlugAndPlayML.

        Args:
            target_column: Target column name (auto-detected if None)
            max_models: Max number of models to try (default: 5)
            prefer_speed: Prefer faster models over accuracy (default: False)
            auto_visualize: Automatically generate visualizations (default: True)
            verbose: Print detailed progress (default: True)
            use_enhanced_loader: Use enhanced loader for misaligned data (default: True)
        """
        self.target_column = target_column
        self.max_models = max_models
        self.prefer_speed = prefer_speed
        self.auto_visualize = auto_visualize
        self.verbose = verbose
        self.use_enhanced_loader = use_enhanced_loader

        # Initialize components
        self.data_analyzer = DataAnalyzer()
        self.problem_classifier = ProblemClassifier()
        self.preprocessor = AutoPreprocessor()
        self.matcher = SmartMatcher(max_models=max_models, prefer_speed=prefer_speed)
        self.enhanced_loader = EnhancedDataLoader(auto_fix=True, verbose=verbose) if use_enhanced_loader else None

        # Storage for results
        self.raw_data = None
        self.loading_metadata = None
        self.characteristics: Optional[DataCharacteristics] = None
        self.problem_definition: Optional[ProblemDefinition] = None
        self.pipeline_recommendation: Optional[PipelineRecommendation] = None
        self.preprocessed_data = None
        self.results = {}

    def run(
        self,
        data: Union[str, Path, pd.DataFrame],
        target_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the complete plug-and-play pipeline.

        This is the ONE method you need to call!

        Args:
            data: CSV file path or DataFrame
            target_column: Optional target column (auto-detected if None)

        Returns:
            Dictionary with complete results
        """
        # Override target if provided
        if target_column:
            self.target_column = target_column

        self._print_header("🚀 Plug-and-Play ML/DL System Starting...")

        # Step 1: Load data
        self._print_step("1. Loading Data")
        self.raw_data = self._load_data(data)
        self._print_success(f"Loaded {len(self.raw_data)} rows × {len(self.raw_data.columns)} columns")

        # Step 2: Analyze data
        self._print_step("2. Analyzing Dataset")
        self.characteristics = self.data_analyzer.analyze(
            self.raw_data,
            target_column=self.target_column
        )
        if self.verbose:
            print(self.data_analyzer.get_summary())

        # Update target column if auto-detected
        if self.target_column is None:
            self.target_column = self.characteristics.target_column

        # Step 3: Classify problem
        self._print_step("3. Classifying Problem Type")
        self.problem_definition = self.problem_classifier.classify(self.characteristics)
        if self.verbose:
            print(self.problem_classifier.get_summary())

        # Step 4: Match to optimal models
        self._print_step("4. Matching to Optimal Models")
        self.pipeline_recommendation = self.matcher.match(
            self.characteristics,
            self.problem_definition
        )
        if self.verbose:
            print(self.matcher.get_summary(self.pipeline_recommendation))

        # Step 5: Preprocess data
        self._print_step("5. Preprocessing Data")
        X_processed, y_processed = self.preprocessor.fit_transform(
            self.raw_data,
            self.characteristics,
            self.problem_definition,
            self.target_column
        )
        if self.verbose:
            print(self.preprocessor.get_summary())

        # Step 6: Train models (placeholder for now)
        self._print_step("6. Training Models")
        self._print_info("Model training implementation coming next...")
        # This will be implemented with the model trainers

        # Step 7: Evaluate and compare
        self._print_step("7. Evaluating Results")
        self._print_info("Evaluation implementation coming next...")

        # Step 8: Generate visualizations
        if self.auto_visualize:
            self._print_step("8. Generating Visualizations")
            self._print_info("Visualization implementation coming next...")

        # Compile results
        self.results = {
            'data_characteristics': self.characteristics,
            'problem_definition': self.problem_definition,
            'pipeline_recommendation': self.pipeline_recommendation,
            'preprocessed_data': {
                'X': X_processed,
                'y': y_processed
            },
            'preprocessing_info': self.preprocessor.get_feature_info(),
            'status': 'analysis_complete',
            'next_steps': [
                'Model training will begin automatically',
                'Results will be saved to ./results/',
                'Visualizations will be saved to ./visualizations/'
            ]
        }

        self._print_header("✅ Analysis Complete!")
        self._print_success("Your data is ready for intelligent model training.")

        return self.results

    def _load_data(self, data: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """Load data from various sources with enhanced error handling."""
        if self.use_enhanced_loader and self.enhanced_loader:
            # Use enhanced loader for robust handling
            df, metadata = self.enhanced_loader.load(data, self.target_column)
            self.loading_metadata = metadata

            # Update target column if it was renamed during normalization
            if (self.target_column and 'column_mapping' in metadata and
                self.target_column in metadata['column_mapping']):
                new_target = metadata['column_mapping'][self.target_column]
                if self.verbose:
                    self._print_info(
                        f"Target column renamed: '{self.target_column}' → '{new_target}'"
                    )
                self.target_column = new_target

            return df
        else:
            # Use basic loader (legacy behavior)
            if isinstance(data, pd.DataFrame):
                return data.copy()

            # Handle file path
            file_path = Path(data)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load based on extension
            ext = file_path.suffix.lower()
            if ext == '.csv':
                return pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif ext == '.json':
                return pd.read_json(file_path)
            elif ext == '.parquet':
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")

    def analyze_only(self, data: Union[str, Path, pd.DataFrame]) -> Dict[str, Any]:
        """
        Only analyze data without training models.

        Useful for quick exploration.

        Args:
            data: CSV file path or DataFrame

        Returns:
            Analysis results
        """
        self.raw_data = self._load_data(data)

        self.characteristics = self.data_analyzer.analyze(
            self.raw_data,
            target_column=self.target_column
        )

        self.problem_definition = self.problem_classifier.classify(self.characteristics)

        return {
            'characteristics': self.characteristics,
            'problem': self.problem_definition,
            'summary': self.get_analysis_summary()
        }

    def get_analysis_summary(self) -> str:
        """Get complete analysis summary."""
        if not self.characteristics or not self.problem_definition:
            return "No analysis performed yet. Call run() or analyze_only() first."

        summary = f"""
{'=' * 60}
PLUG-AND-PLAY ML/DL SYSTEM - ANALYSIS REPORT
{'=' * 60}

{self.data_analyzer.get_summary()}

{self.problem_classifier.get_summary()}
"""
        if self.pipeline_recommendation:
            summary += f"\n{self.matcher.get_summary(self.pipeline_recommendation)}"

        if self.preprocessor.is_fitted:
            summary += f"\n{self.preprocessor.get_summary()}"

        return summary

    def get_recommendations(self) -> str:
        """Get actionable recommendations."""
        if not self.problem_definition:
            return "Run analysis first to get recommendations."

        recommendations = ["🎯 RECOMMENDATIONS:\n"]

        # Data quality recommendations
        if self.characteristics.missing_percentage > 20:
            recommendations.append(
                "⚠️  Consider collecting more complete data if possible"
            )

        if self.characteristics.n_samples < 1000:
            recommendations.append(
                "💡 Small dataset detected - consider data augmentation"
            )

        # Model recommendations
        if self.pipeline_recommendation:
            top_model = self.pipeline_recommendation.models[0]
            recommendations.append(
                f"✅ Best model: {top_model.model_name} "
                f"(confidence: {top_model.confidence:.0%})"
            )

        # Feature engineering
        if self.problem_definition.requires_feature_engineering:
            recommendations.append(
                "🔧 Feature engineering will significantly improve results"
            )

        # Special considerations
        for note in self.problem_definition.special_notes:
            recommendations.append(note)

        return "\n".join(recommendations)

    def save_results(self, output_dir: str = "./results"):
        """Save all results to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save analysis report
        report_path = output_path / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(self.get_analysis_summary())

        self._print_success(f"Results saved to {output_dir}/")

    # Helper methods for pretty printing
    def _print_header(self, text: str):
        """Print section header."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"{text:^60}")
            print(f"{'=' * 60}\n")

    def _print_step(self, text: str):
        """Print step header."""
        if self.verbose:
            print(f"\n{'─' * 60}")
            print(f"📍 {text}")
            print(f"{'─' * 60}")

    def _print_success(self, text: str):
        """Print success message."""
        if self.verbose:
            print(f"✅ {text}")

    def _print_info(self, text: str):
        """Print info message."""
        if self.verbose:
            print(f"ℹ️  {text}")

    def _print_warning(self, text: str):
        """Print warning message."""
        if self.verbose:
            print(f"⚠️  {text}")

    def __repr__(self) -> str:
        """String representation."""
        status = "Not run yet"
        if self.results:
            status = f"Analysis complete - {self.problem_definition.problem_name}"

        return f"PlugAndPlayML(status='{status}')"


# Convenience function for quick usage
def auto_ml(data: Union[str, Path, pd.DataFrame], **kwargs) -> Dict[str, Any]:
    """
    Quick convenience function for plug-and-play ML.

    Usage:
        results = auto_ml("data.csv")

    Args:
        data: CSV file path or DataFrame
        **kwargs: Additional arguments for PlugAndPlayML

    Returns:
        Complete results dictionary
    """
    model = PlugAndPlayML(**kwargs)
    return model.run(data)
