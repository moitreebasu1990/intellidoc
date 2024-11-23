# Copyright (c) Moitree Basu under MIT License.
# Advanced LLM-based document classification evaluation
# Code: https://github.com/moitreebasu1990/intellidoc

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from intellidoc.data_processing.dataset import TextClassificationDataset
from intellidoc.utilities.save_load_model import load_model


def compute_metrics(
    predictions: List[int], labels: List[int], label_map: Optional[Dict] = None
) -> Dict:
    """
    Compute various metrics for model evaluation.

    Args:
        predictions (List[int]): Model predictions
        labels (List[int]): True labels
        label_map (Optional[Dict]): Mapping of label indices to label names

    Returns:
        Dict: Dictionary containing various metrics
    """
    report = classification_report(labels, predictions, output_dict=True)

    if label_map:
        # Convert numerical labels to their string representations
        mapped_report = {}
        for key in report:
            if key.isdigit():
                mapped_report[label_map[int(key)]] = report[key]
            else:
                mapped_report[key] = report[key]
        return mapped_report

    return report


def eval_llm_classification(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: AutoTokenizer,
    model_path: str = "model.pth",
    label_map: Optional[Dict] = None,
    output_path: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
) -> Dict:
    """
    Evaluate an LLM-based classification model with detailed metrics and analysis.

    Args:
        model (nn.Module): The PyTorch model to evaluate
        val_loader (DataLoader): DataLoader containing validation data
        device (torch.device): Device to use for evaluation
        tokenizer (AutoTokenizer): Tokenizer for processing text
        model_path (str): Path to saved model checkpoint
        label_map (Optional[Dict]): Mapping of label indices to label names
        output_path (Optional[str]): Path to save evaluation results
        batch_size (int): Batch size for evaluation
        max_length (int): Maximum sequence length

    Returns:
        Dict: Evaluation metrics and results
    """
    # Load model from checkpoint if provided
    if model_path and os.path.exists(model_path):
        model = load_model(model_path, "model.pth")
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return {}

    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    all_confidences = []
    incorrect_examples = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["doc_labels"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            # Get predictions and confidences
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predictions = torch.max(probabilities, dim=-1)

            # Store predictions, labels, and confidences
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().tolist())
            all_confidences.extend(confidence.cpu().tolist())

            # Track incorrect predictions
            incorrect_mask = predictions != labels.squeeze(-1)
            if incorrect_mask.any():
                incorrect_indices = torch.where(incorrect_mask)[0]
                for idx in incorrect_indices:
                    incorrect_examples.append(
                        {
                            "text": tokenizer.decode(
                                input_ids[idx], skip_special_tokens=True
                            ),
                            "true_label": labels[idx].item(),
                            "predicted_label": predictions[idx].item(),
                            "confidence": confidence[idx].item(),
                        }
                    )

    # Compute metrics
    metrics = compute_metrics(all_predictions, all_labels, label_map)

    # Add additional analysis
    evaluation_results = {
        "metrics": metrics,
        "average_confidence": sum(all_confidences) / len(all_confidences),
        "incorrect_predictions": incorrect_examples[
            :10
        ],  # Show first 10 incorrect predictions
        "confidence_distribution": {
            "min": min(all_confidences),
            "max": max(all_confidences),
            "mean": sum(all_confidences) / len(all_confidences),
        },
    }

    # Save results if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"Evaluation results saved to {output_path}")

    return evaluation_results


def eval():
    # Set device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load validation data
    val_df = pd.read_csv("data/processed/val_data.csv")

    # Define label mapping
    label_map = {i: f"class_{i}" for i in range(10)}  # Modify based on your classes

    # Initialize tokenizer and model
    model_name = "roberta-base"  # or your preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_map)
    )

    # Initialize dataset and dataloader
    val_dataset = TextClassificationDataset(val_df["processed_text"], val_df["label"])
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Run evaluation
    results = eval_llm_classification(
        model=model,
        val_loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        model_path="models/llm_classifier.pth",
        label_map=label_map,
        output_path="evaluation_results/llm_evaluation.json",
    )

    # Print summary metrics
    print("\nEvaluation Results:")
    print(f"Macro Avg F1-Score: {results['metrics']['macro avg']['f1-score']:.4f}")
    print(
        f"Weighted Avg F1-Score: {results['metrics']['weighted avg']['f1-score']:.4f}"
    )
    print(f"Average Confidence: {results['average_confidence']:.4f}")


if __name__ == "__main__":
    eval()
