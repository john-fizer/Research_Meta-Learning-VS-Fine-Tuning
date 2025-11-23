"""
Fine-Tuning Data Formatter

Converts dataset samples into format suitable for fine-tuning.
"""

from typing import List, Dict, Any


class FineTuningDataFormatter:
    """
    Format data for fine-tuning

    Converts dataset samples into training format:
    - Text: Input text for the model
    - Label: Target label (int for classification)
    """

    @staticmethod
    def format_classification(
        samples: List[Dict[str, Any]],
        text_key: str = "input",
        label_key: str = "expected"
    ) -> List[Dict[str, Any]]:
        """
        Format samples for classification fine-tuning

        Args:
            samples: Dataset samples
            text_key: Key for input text
            label_key: Key for label

        Returns:
            Formatted samples with 'text' and 'label' keys
        """
        formatted = []

        for sample in samples:
            # Extract text
            text = sample.get(text_key, "")
            if isinstance(text, dict):
                # Handle complex input structures
                text = str(text)

            # Extract label
            label = sample.get(label_key)

            # Convert label to int if string
            if isinstance(label, str):
                # For binary classification
                if label.lower() in ['positive', 'pos', '1', 'true', 'yes']:
                    label = 1
                elif label.lower() in ['negative', 'neg', '0', 'false', 'no']:
                    label = 0
                else:
                    # For multi-class, try to parse as int
                    try:
                        label = int(label)
                    except ValueError:
                        # Use hash if can't convert
                        label = hash(label) % 10

            formatted.append({
                "text": str(text),
                "label": int(label)
            })

        return formatted

    @staticmethod
    def format_commonsense_qa(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format CommonsenseQA samples for fine-tuning

        Args:
            samples: CommonsenseQA samples

        Returns:
            Formatted samples
        """
        formatted = []

        for sample in samples:
            # Create text from question and choices
            question = sample.get('question', '')
            choices = sample.get('choices', {})
            choice_text = choices.get('text', [])

            # Format as: "Question: {q} Choices: A) {a} B) {b} ..."
            text = f"Question: {question} Choices: "
            for i, choice in enumerate(choice_text):
                label = chr(65 + i)  # A, B, C, ...
                text += f"{label}) {choice} "

            # Get correct answer label (A, B, C, etc.) and convert to index
            answer_key = sample.get('answerKey', 'A')
            label = ord(answer_key) - 65  # Convert A->0, B->1, etc.

            formatted.append({
                "text": text.strip(),
                "label": label
            })

        return formatted

    @staticmethod
    def format_sentiment140(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format Sentiment140 samples for fine-tuning

        Args:
            samples: Sentiment140 samples

        Returns:
            Formatted samples
        """
        formatted = []

        for sample in samples:
            text = sample.get('text', '')
            sentiment = sample.get('sentiment', 0)

            # Convert sentiment to binary (0=negative, 1=positive)
            # Sentiment140 uses 0=negative, 4=positive
            label = 1 if sentiment > 2 else 0

            formatted.append({
                "text": text,
                "label": label
            })

        return formatted

    @staticmethod
    def split_train_eval(
        samples: List[Dict[str, Any]],
        eval_ratio: float = 0.2
    ) -> tuple:
        """
        Split data into train and eval sets

        Args:
            samples: All samples
            eval_ratio: Ratio for evaluation set

        Returns:
            (train_samples, eval_samples)
        """
        split_idx = int(len(samples) * (1 - eval_ratio))
        return samples[:split_idx], samples[split_idx:]
