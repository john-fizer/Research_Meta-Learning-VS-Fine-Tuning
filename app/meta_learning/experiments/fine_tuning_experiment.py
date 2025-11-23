"""
Fine-Tuning Experiment

Integrates fine-tuning into the experiment framework for comparison with meta-prompting.
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from ..fine_tuning import FineTuningTrainer, FineTuningDataFormatter, ModelManager
from ..fine_tuning.trainer import FineTuningConfig
from .runner import ExperimentConfig, ExperimentResult


class FineTuningExperiment:
    """
    Run fine-tuning experiments for comparison with meta-prompting

    Provides fair comparison:
    - Same datasets
    - Same evaluation metrics
    - Same sample sizes
    - Comparable model complexity
    """

    def __init__(
        self,
        save_path: Optional[Path] = None,
        use_gpu: bool = True
    ):
        self.save_path = save_path or Path("./data/meta_learning/experiments")
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.model_manager = ModelManager(save_path=self.save_path / "fine_tuning")
        self.use_gpu = use_gpu

    async def run_fine_tuning_experiment(
        self,
        config: ExperimentConfig,
        dataset_loader,
        dataset_name: str
    ) -> ExperimentResult:
        """
        Run fine-tuning experiment

        Args:
            config: Experiment configuration
            dataset_loader: Dataset loader function
            dataset_name: Name of dataset (for formatting)

        Returns:
            Experiment results
        """
        print(f"\n{'='*70}")
        print(f"RUNNING FINE-TUNING EXPERIMENT: {config.name}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}\n")

        # Load full dataset for training
        all_samples = dataset_loader(config.sample_size)

        # Format data for fine-tuning
        formatter = FineTuningDataFormatter()

        if dataset_name == "commonsense_qa":
            formatted_samples = formatter.format_commonsense_qa(all_samples)
            num_labels = 5  # A, B, C, D, E
        elif dataset_name == "sentiment140":
            formatted_samples = formatter.format_sentiment140(all_samples)
            num_labels = 2  # Positive, Negative
        else:
            formatted_samples = formatter.format_classification(all_samples)
            num_labels = 2  # Default binary

        # Split train/eval
        train_samples, eval_samples = formatter.split_train_eval(
            formatted_samples,
            eval_ratio=0.2
        )

        print(f"Training samples: {len(train_samples)}")
        print(f"Evaluation samples: {len(eval_samples)}\n")

        # Get recommended model
        model_name = self.model_manager.get_recommended_model(
            task_type="classification",
            size="small"
        )

        # Configure fine-tuning
        ft_config = FineTuningConfig(
            model_name=model_name,
            task_type="classification",
            num_labels=num_labels,
            num_epochs=3,
            batch_size=16,
            use_gpu=self.use_gpu,
        )

        # Initialize trainer
        trainer = FineTuningTrainer(
            config=ft_config,
            save_path=self.save_path / "fine_tuning"
        )

        # Train model
        train_results = trainer.train(
            train_samples=train_samples,
            eval_samples=eval_samples,
            experiment_name=config.name
        )

        # Evaluate on test set (use eval samples as test for now)
        final_metrics = trainer.evaluate_model(eval_samples)

        # Register model
        self.model_manager.register_model(
            experiment_name=config.name,
            model_path=Path(train_results['model_path']),
            metrics=final_metrics,
            config=asdict(ft_config)
        )

        # Create experiment result
        experiment_result = ExperimentResult(
            config=config,
            final_performance=final_metrics,
            performance_history=[final_metrics],  # Fine-tuning doesn't track iteration history
            best_performance=final_metrics,
            improvement=0.0,  # No iterative improvement in standard fine-tuning
            convergence_iteration=ft_config.num_epochs,
            timestamp=datetime.now(),
            metadata={
                'approach': 'fine_tuning',
                'model_name': model_name,
                'train_loss': train_results['train_loss'],
                'train_runtime': train_results['train_runtime'],
                'model_path': train_results['model_path'],
            }
        )

        print(f"\n{'='*70}")
        print("FINE-TUNING COMPLETE")
        print(f"Final Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"F1 Score: {final_metrics['f1']:.4f}")
        print(f"{'='*70}\n")

        return experiment_result

    async def run_full_comparison(
        self,
        dataset_name: str,
        dataset_loader,
        initial_prompt: str,
        meta_prompting_runner,
        num_iterations: int = 10,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run complete comparison: meta-prompting vs fine-tuning

        Args:
            dataset_name: Dataset name
            dataset_loader: Dataset loader function
            initial_prompt: Initial prompt for meta-prompting
            meta_prompting_runner: ExperimentRunner instance
            num_iterations: Iterations for meta-prompting
            sample_size: Sample size

        Returns:
            Comparison results
        """
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE COMPARISON: META-PROMPTING VS FINE-TUNING")
        print(f"Dataset: {dataset_name}")
        print(f"Sample Size: {sample_size}")
        print(f"{'='*80}\n")

        # Configure experiments
        meta_config = ExperimentConfig(
            name=f"{dataset_name}_meta_prompting_v2",
            description="Meta-prompting with adaptive curriculum learning",
            dataset_name=dataset_name,
            approach="meta_prompting",
            num_iterations=num_iterations,
            sample_size=sample_size
        )

        ft_config = ExperimentConfig(
            name=f"{dataset_name}_fine_tuning",
            description="Fine-tuned language model",
            dataset_name=dataset_name,
            approach="fine_tuning",
            num_iterations=1,  # Fine-tuning doesn't iterate
            sample_size=sample_size
        )

        # Run meta-prompting experiment
        print("\n" + "="*80)
        print("EXPERIMENT 1: META-PROMPTING")
        print("="*80)
        meta_result = await meta_prompting_runner.run_meta_prompting_experiment(
            meta_config,
            dataset_loader,
            initial_prompt
        )

        # Run fine-tuning experiment
        print("\n" + "="*80)
        print("EXPERIMENT 2: FINE-TUNING")
        print("="*80)
        ft_result = await self.run_fine_tuning_experiment(
            ft_config,
            dataset_loader,
            dataset_name
        )

        # Compare results
        comparison = self._compare_results(meta_result, ft_result)

        # Print summary
        self._print_comparison_summary(comparison, dataset_name)

        return comparison

    def _compare_results(
        self,
        meta_result: ExperimentResult,
        ft_result: ExperimentResult
    ) -> Dict[str, Any]:
        """Compare meta-prompting and fine-tuning results"""

        meta_acc = meta_result.final_performance.get('accuracy', 0)
        ft_acc = ft_result.final_performance.get('accuracy', 0)

        meta_f1 = meta_result.final_performance.get('f1', 0)
        ft_f1 = ft_result.final_performance.get('f1', 0)

        return {
            "meta_prompting": {
                "accuracy": meta_acc,
                "f1": meta_f1,
                "improvement": meta_result.improvement,
                "convergence_iteration": meta_result.convergence_iteration,
            },
            "fine_tuning": {
                "accuracy": ft_acc,
                "f1": ft_f1,
                "train_loss": ft_result.metadata.get('train_loss'),
                "model": ft_result.metadata.get('model_name'),
            },
            "winner": "meta_prompting" if meta_acc > ft_acc else "fine_tuning",
            "accuracy_difference": abs(meta_acc - ft_acc),
            "f1_difference": abs(meta_f1 - ft_f1),
            "meta_prompting_wins": meta_acc > ft_acc,
        }

    def _print_comparison_summary(self, comparison: Dict[str, Any], dataset_name: str):
        """Print comparison summary"""
        print(f"\n{'='*80}")
        print(f"FINAL COMPARISON SUMMARY: {dataset_name}")
        print(f"{'='*80}\n")

        meta = comparison["meta_prompting"]
        ft = comparison["fine_tuning"]

        print("META-PROMPTING RESULTS:")
        print(f"  Accuracy: {meta['accuracy']:.4f}")
        print(f"  F1 Score: {meta['f1']:.4f}")
        print(f"  Improvement: {meta['improvement']:.4f}")
        print(f"  Convergence: {meta['convergence_iteration']} iterations")

        print("\nFINE-TUNING RESULTS:")
        print(f"  Accuracy: {ft['accuracy']:.4f}")
        print(f"  F1 Score: {ft['f1']:.4f}")
        print(f"  Model: {ft['model']}")
        print(f"  Train Loss: {ft['train_loss']:.4f}")

        print(f"\n{'='*80}")
        print("CONCLUSION:")
        if comparison["meta_prompting_wins"]:
            print("  ✓ META-PROMPTING WINS!")
            print(f"    Accuracy advantage: {comparison['accuracy_difference']:.4f}")
        else:
            print("  ✓ FINE-TUNING WINS!")
            print(f"    Accuracy advantage: {comparison['accuracy_difference']:.4f}")

        print(f"\n{'='*80}\n")
