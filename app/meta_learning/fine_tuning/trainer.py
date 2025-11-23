"""
Fine-Tuning Trainer

Handles training small language models on task-specific data.
Supports multiple model architectures and efficient training strategies.
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning"""
    model_name: str = "distilbert-base-uncased"  # Small, fast model
    task_type: str = "classification"  # or "generation"
    num_labels: int = 2
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 16
    max_length: int = 128
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_path: Optional[Path] = None
    use_gpu: bool = True


class FineTuningTrainer:
    """
    Fine-tune small language models for task-specific performance

    Supports:
    - Classification (DistilBERT, BERT, RoBERTa)
    - Generation (GPT-2, TinyLlama)
    - Multiple training strategies

    Research Question: How does fine-tuning compare to meta-prompting?
    """

    def __init__(
        self,
        config: FineTuningConfig,
        save_path: Optional[Path] = None
    ):
        """
        Initialize fine-tuning trainer

        Args:
            config: Fine-tuning configuration
            save_path: Path to save models and checkpoints
        """
        self.config = config
        self.save_path = save_path or Path("./data/meta_learning/fine_tuning")
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.use_gpu else "cpu"
        )

        print(f"Using device: {self.device}")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize model
        self.model = None
        self.trainer = None

    def prepare_data(
        self,
        train_samples: List[Dict[str, Any]],
        eval_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare data for fine-tuning

        Args:
            train_samples: Training samples with 'text' and 'label' keys
            eval_samples: Optional evaluation samples

        Returns:
            Tokenized train and eval datasets
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_length
            )

        # Convert to HuggingFace datasets
        from datasets import Dataset

        train_dict = {
            "text": [s["text"] for s in train_samples],
            "label": [s["label"] for s in train_samples]
        }
        train_dataset = Dataset.from_dict(train_dict)
        train_dataset = train_dataset.map(tokenize_function, batched=True)

        eval_dataset = None
        if eval_samples:
            eval_dict = {
                "text": [s["text"] for s in eval_samples],
                "label": [s["label"] for s in eval_samples]
            }
            eval_dataset = Dataset.from_dict(eval_dict)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        return train_dataset, eval_dataset

    def train(
        self,
        train_samples: List[Dict[str, Any]],
        eval_samples: Optional[List[Dict[str, Any]]] = None,
        experiment_name: str = "fine_tuning"
    ) -> Dict[str, Any]:
        """
        Fine-tune model on training data

        Args:
            train_samples: Training samples
            eval_samples: Optional evaluation samples
            experiment_name: Name for this experiment

        Returns:
            Training results and metrics
        """
        print(f"\n{'='*70}")
        print(f"FINE-TUNING: {experiment_name}")
        print(f"Model: {self.config.model_name}")
        print(f"Samples: {len(train_samples)} train, {len(eval_samples) if eval_samples else 0} eval")
        print(f"{'='*70}\n")

        # Prepare data
        train_dataset, eval_dataset = self.prepare_data(train_samples, eval_samples)

        # Initialize model
        if self.config.task_type == "classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name
            )

        self.model.to(self.device)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.save_path / experiment_name),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_dir=str(self.save_path / experiment_name / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        # Train
        print("Training...")
        train_result = self.trainer.train()

        # Evaluate
        if eval_dataset:
            print("\nEvaluating...")
            eval_result = self.trainer.evaluate()
        else:
            eval_result = {}

        # Save model
        model_path = self.save_path / experiment_name / "final_model"
        self.trainer.save_model(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))

        # Compile results
        results = {
            "experiment_name": experiment_name,
            "model_name": self.config.model_name,
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples) if eval_samples else 0,
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_steps": train_result.metrics["train_steps_per_second"],
            "eval_loss": eval_result.get("eval_loss"),
            "eval_runtime": eval_result.get("eval_runtime"),
            "model_path": str(model_path),
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        results_path = self.save_path / experiment_name / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nTraining complete!")
        print(f"  Train loss: {results['train_loss']:.4f}")
        if eval_dataset:
            print(f"  Eval loss: {results['eval_loss']:.4f}")
        print(f"  Model saved: {model_path}")

        return results

    def evaluate_model(
        self,
        test_samples: List[Dict[str, Any]],
        model_path: Optional[Path] = None
    ) -> Dict[str, float]:
        """
        Evaluate fine-tuned model

        Args:
            test_samples: Test samples
            model_path: Path to saved model (uses current model if None)

        Returns:
            Evaluation metrics
        """
        # Load model if path provided
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path)
            )
            self.model.to(self.device)

        if self.model is None:
            raise ValueError("No model loaded. Train first or provide model_path.")

        # Prepare data
        test_dataset, _ = self.prepare_data(test_samples)

        # Create temporary trainer for evaluation
        training_args = TrainingArguments(
            output_dir=str(self.save_path / "temp"),
            per_device_eval_batch_size=self.config.batch_size,
        )

        temp_trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
        )

        # Evaluate
        eval_result = temp_trainer.evaluate(test_dataset)

        # Get predictions for detailed metrics
        predictions = temp_trainer.predict(test_dataset)
        pred_labels = predictions.predictions.argmax(-1)
        true_labels = predictions.label_ids

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "eval_loss": eval_result["eval_loss"],
        }

        return metrics

    def predict(
        self,
        texts: List[str],
        model_path: Optional[Path] = None
    ) -> List[int]:
        """
        Make predictions with fine-tuned model

        Args:
            texts: Input texts
            model_path: Path to saved model (uses current model if None)

        Returns:
            Predicted labels
        """
        # Load model if path provided
        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path)
            )
            self.model.to(self.device)

        if self.model is None:
            raise ValueError("No model loaded. Train first or provide model_path.")

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)

        return predictions.cpu().tolist()
