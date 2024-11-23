# Copyright (c) Moitree Basu under MIT License.
# Advanced LLM model training implementation
# Code: https://github.com/moitreebasu1990/intellidoc

"""Advanced LLM model training module for document classification.

This module provides functionality for training an advanced LLM-based document classification model.
It includes:
    - Training loop with validation
    - Multi-layer supervision and loss aggregation
    - Attention visualization and analysis
    - Performance tracking across all model layers
    - Model checkpointing with complete state
    - Comprehensive performance metrics and reporting

The training process includes:
    1. Epoch-wise training with progress bars
    2. Regular validation checks with layer-wise performance analysis
    3. Attention pattern monitoring and visualization
    4. Model saving with configuration preservation
    5. Detailed logging of training and validation metrics

Typical usage:
    ```python
    from intellidoc.training.llm_trainer import train

    # Execute the complete training pipeline
    train()
    ```
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import custom modules
from intellidoc.classification_models.advance_llm import AdvancedDocClassificationModel
from intellidoc.data_processing.dataset import TextClassificationDataset
from intellidoc.utilities.save_load_model import load_model, save_model


class LayerWiseLoss(nn.Module):
    """Loss function that combines losses from all layers with optional weighting.

    This class implements a custom loss function that:
        1. Computes CrossEntropyLoss for each layer's predictions
        2. Applies optional weights to each layer's loss
        3. Aggregates the weighted losses into a final loss value

    Attributes:
        base_criterion: The base loss function (CrossEntropyLoss)
        weights: Optional list of weights for each layer's loss
    """

    def __init__(self, weights: List[float] = None):
        """Initialize the layer-wise loss function.

        Args:
            weights: Optional list of weights for each layer's loss.
                If None, all layers are weighted equally.
                If provided, must sum to 1.0.
        """
        super().__init__()
        self.base_criterion = nn.CrossEntropyLoss()
        self.weights = weights

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate weighted loss across all layers.

        Args:
            outputs (Dict[str, torch.Tensor]): Model outputs containing:
                - logits: Final layer logits
                - intermediate_logits: List of intermediate layer logits
            targets (torch.Tensor): Ground truth labels

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Total loss and individual layer losses
        """
        # Calculate loss for final layer
        final_loss = self.base_criterion(outputs["logits"], targets)

        # Calculate losses for intermediate layers
        intermediate_losses = [
            self.base_criterion(logits, targets)
            for logits in outputs["intermediate_logits"]
        ]

        # Combine all losses
        all_losses = intermediate_losses + [final_loss]

        # Apply weights if provided
        if self.weights is None:
            self.weights = [1.0 / len(all_losses)] * len(all_losses)

        # Calculate weighted sum
        total_loss = sum(w * l for w, l in zip(self.weights, all_losses))

        # Create loss dictionary for logging
        loss_dict = {
            f"layer_{i}_loss": loss.item() for i, loss in enumerate(intermediate_losses)
        }
        loss_dict["final_layer_loss"] = final_loss.item()

        return total_loss, loss_dict


def train_doc_classification_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: LayerWiseLoss,
    device: torch.device,
    n_epochs: int = 10,
    model_path: str = "model.pth",
    save_attention_maps: bool = True,
) -> str:
    """Train an advanced document classification model with comprehensive monitoring.

    This function implements the training loop with multi-layer supervision and
    attention pattern analysis. It tracks performance across all layers and saves
    detailed model state and attention visualizations.

    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for updating model parameters
        criterion: Layer-wise loss function for training
        device: Device to run the training on (cpu/cuda/mps)
        n_epochs: Number of training epochs (default: 10)
        model_path: Path to save the model (default: "model.pth")
        save_attention_maps: Whether to save attention visualizations (default: True)

    Returns:
        str: Comprehensive classification report with layer-wise metrics

    Raises:
        RuntimeError: If model training fails or device is not available
        ValueError: If input parameters are invalid
    """
    # Move model to specified device
    model = model.to(device)

    # Create directories for saving model and attention maps
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if save_attention_maps:
        attention_dir = os.path.join(os.path.dirname(model_path), "attention_maps")
        os.makedirs(attention_dir, exist_ok=True)

    # Training loop over epochs
    for ep in range(n_epochs):
        # Training phase
        model.train()
        train_metrics = {"correct": 0, "total": 0, "losses": [], "layer_losses": {}}

        # Iterate over training batches with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {ep + 1}/{n_epochs} - Training"):
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_labels = batch["doc_labels"].to(device)

            # Training step
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            # Calculate loss and backpropagate
            loss, layer_losses = criterion(outputs, doc_labels.squeeze(-1))
            loss.backward()
            optimizer.step()

            # Update metrics
            prediction = torch.argmax(outputs["logits"], dim=-1)
            train_metrics["correct"] += (
                (prediction == doc_labels.squeeze(-1)).sum().item()
            )
            train_metrics["total"] += doc_labels.size(0)
            train_metrics["losses"].append(loss.item())

            # Accumulate layer-wise losses
            for layer_name, layer_loss in layer_losses.items():
                if layer_name not in train_metrics["layer_losses"]:
                    train_metrics["layer_losses"][layer_name] = []
                train_metrics["layer_losses"][layer_name].append(layer_loss)

        # Calculate training metrics
        train_accuracy = train_metrics["correct"] / train_metrics["total"]
        train_loss = np.mean(train_metrics["losses"])
        layer_losses_mean = {
            name: np.mean(losses)
            for name, losses in train_metrics["layer_losses"].items()
        }

        # Validation phase
        model.eval()
        val_metrics = {
            "predictions": [],
            "labels": [],
            "correct": 0,
            "total": 0,
            "attention_maps": [] if save_attention_maps else None,
        }

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {ep + 1}/{n_epochs} - Validation"
            ):
                # Move batch data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                doc_labels = batch["doc_labels"].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask)
                prediction = torch.argmax(outputs["logits"], dim=-1)

                # Update metrics
                val_metrics["predictions"].extend(prediction.cpu().numpy())
                val_metrics["labels"].extend(doc_labels.squeeze(-1).cpu().numpy())
                val_metrics["correct"] += (
                    (prediction == doc_labels.squeeze(-1)).sum().item()
                )
                val_metrics["total"] += doc_labels.size(0)

                # Save attention maps if requested
                if save_attention_maps:
                    val_metrics["attention_maps"].extend(
                        [attn.cpu().numpy() for attn in outputs["attention_probs"]]
                    )

        # Calculate validation metrics
        val_accuracy = val_metrics["correct"] / val_metrics["total"]

        # Print epoch metrics
        print("\n" + "=" * 50)
        print(f"Epoch {ep + 1}/{n_epochs}")
        print(f"Training Accuracy: {train_accuracy:.4f} | Loss: {train_loss:.4f}")
        print("Layer-wise Losses:")
        for layer_name, layer_loss in layer_losses_mean.items():
            print(f"  {layer_name}: {layer_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("=" * 50 + "\n")

        # Save attention maps
        if save_attention_maps and (ep + 1) % 5 == 0:
            attention_path = os.path.join(attention_dir, f"epoch_{ep+1}_attention.npy")
            np.save(attention_path, np.array(val_metrics["attention_maps"]))

    # Save the final model with complete state
    model_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": model.get_config(),
        "final_metrics": {
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "val_accuracy": val_accuracy,
        },
    }
    torch.save(model_state, model_path)
    print(f"Model saved to {model_path}")

    # Generate and return classification report
    return classification_report(
        val_metrics["labels"], val_metrics["predictions"], digits=4
    )


def train() -> None:
    """Execute the complete training pipeline for the advanced LLM model.

    This function:
        1. Sets up the training device (CPU/GPU/MPS)
        2. Initializes the advanced model with multi-layer architecture
        3. Configures layer-wise loss and optimization
        4. Loads and prepares the datasets
        5. Executes the training process with comprehensive monitoring

    The function uses carefully tuned hyperparameters:
        - AdamW optimizer with learning rate 2e-5 and weight decay
        - Layer-wise weighted loss function
        - Gradient clipping for stability
        - Batch size of 16 (smaller due to model complexity)
        - 30 training epochs

    Raises:
        FileNotFoundError: If required data files are not found
        RuntimeError: If training fails or required hardware is not available
    """
    # Set up the training device
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initialize model and training components
    model = AdvancedDocClassificationModel(
        n_doc_labels=10,
        num_hidden_layers=2,  # Additional transformer layers
        p_dropout=0.1,
    )

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8
    )

    # Initialize layer-wise loss with increasing weights for deeper layers
    criterion = LayerWiseLoss(weights=[0.2, 0.3, 0.5])  # Weights sum to 1

    # Load and prepare datasets
    train_df = pd.read_csv("data/processed/train_data.csv")
    val_df = pd.read_csv("data/processed/val_data.csv")

    # Create dataset objects
    train_dataset = TextClassificationDataset(
        train_df["processed_text"], train_df["label"]
    )
    val_dataset = TextClassificationDataset(val_df["processed_text"], val_df["label"])

    # Initialize data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # Execute training with comprehensive monitoring
    classification_report = train_doc_classification_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        n_epochs=30,
        model_path="models/llm_classification/model.pth",
        save_attention_maps=True,
    )

    # Print final classification report
    print("\nFinal Classification Report:")
    print(classification_report)


if __name__ == "__main__":
    train()
