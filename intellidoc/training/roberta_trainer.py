# Copyright (c) Moitree Basu under MIT License.
# RoBERTa model training implementation
# Code: https://github.com/moitreebasu1990/intellidoc

"""RoBERTa model training module for document classification.

This module provides functionality for training a RoBERTa-based document classification model.
It includes:
    - Training loop with validation
    - Accuracy tracking for both training and validation
    - Model checkpointing
    - Performance reporting using classification metrics

The training process includes:
    1. Epoch-wise training with progress bars
    2. Regular validation checks
    3. Model saving after training
    4. Detailed logging of training and validation metrics

Typical usage:
    ```python
    from intellidoc.training.roberta_trainer import train

    # Execute the complete training pipeline
    train()
    ```

Note:
    This module assumes the input data is preprocessed and properly formatted.
    See data_processing module for data preparation steps.
"""

import os

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import custom modules
from intellidoc.classification_models.roberta import RobertaDocClassificationModel
from intellidoc.data_processing.dataset import TextClassificationDataset
from intellidoc.utilities.save_load_model import load_model, save_model


def train_doc_classification_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    n_epochs: int = 10,
    model_path: str = "model.pth",
) -> str:
    """Train a document classification model with validation monitoring.

    This function implements the training loop for the document classification model.
    It performs both training and validation for the specified number of epochs,
    tracks metrics, and saves the model after training.

    The training process includes:
        1. Moving data to the specified device
        2. Forward pass through the model
        3. Loss computation and backpropagation
        4. Model parameter updates
        5. Validation on a separate dataset
        6. Metric tracking and reporting

    Args:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for updating model parameters
        criterion: Loss function for training
        device: Device to run the training on (cpu/cuda/mps)
        n_epochs: Number of training epochs (default: 10)
        model_path: Path to save the trained model (default: "model.pth")

    Returns:
        str: Classification report containing precision, recall, and F1-score metrics

    Raises:
        RuntimeError: If training fails or device is not available
        ValueError: If input parameters are invalid
    """
    # Move model to specified device (CPU/GPU/MPS)
    model = model.to(device)

    # Training loop over epochs
    for ep in range(n_epochs):
        # Training phase
        model.train()
        correct_train = 0
        total_train = 0

        # Iterate over training batches with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {ep + 1}/{n_epochs} - Training"):
            # Move batch data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_labels = batch["doc_labels"].to(device)

            # Training step
            optimizer.zero_grad()
            model_outputs = model(input_ids, attention_mask)
            train_loss = criterion(model_outputs, doc_labels.squeeze(-1))
            train_loss.backward()
            optimizer.step()

            # Calculate training accuracy
            prediction = torch.argmax(model_outputs, dim=-1)
            correct_train += (prediction == doc_labels.squeeze(-1)).sum().item()
            total_train += doc_labels.size(0)

        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval()
        all_predictions = []
        all_labels = []
        correct_val = 0
        total_val = 0

        # Disable gradient computation for validation
        with torch.no_grad():
            # Iterate over validation batches with progress bar
            for batch in tqdm(
                val_loader, desc=f"Epoch {ep + 1}/{n_epochs} - Validation"
            ):
                # Move batch data to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                doc_labels = batch["doc_labels"].to(device)

                # Forward pass
                model_outputs = model(input_ids, attention_mask)
                prediction = torch.argmax(model_outputs, dim=-1)

                # Collect predictions and labels for metrics
                all_predictions.extend(prediction.cpu().numpy())
                all_labels.extend(doc_labels.squeeze(-1).cpu().numpy())

                # Calculate validation accuracy
                correct_val += (prediction == doc_labels.squeeze(-1)).sum().item()
                total_val += doc_labels.size(0)

            # Print epoch metrics
            val_accuracy = correct_val / total_val
            print("\n----------------------------")
            print(
                f"Epoch {ep + 1}/{n_epochs} - Training Accuracy: {train_accuracy:.4f} | Validation Accuracy: {val_accuracy:.4f}"
            )
            print("----------------------------\n")

    # Save the final model
    save_model(model, model_path, "model.pth")
    print(f"Model saved to {model_path}")

    # Return classification report
    return classification_report(all_labels, all_predictions)


def train() -> None:
    """Execute the complete training pipeline.

    This function:
        1. Sets up the training device (CPU/GPU/MPS)
        2. Initializes the model, optimizer, and loss function
        3. Loads and prepares the datasets
        4. Executes the training process

    The function uses default hyperparameters suitable for document classification:
        - AdamW optimizer with learning rate 3e-5
        - CrossEntropyLoss as the loss function
        - Batch size of 32
        - 40 training epochs

    Device selection priority:
        1. CUDA GPU if available
        2. Apple Silicon MPS if available
        3. CPU as fallback

    Data loading:
        - Training data from "data/processed/train.csv"
        - Validation data from "data/processed/val.csv"

    Model configuration:
        - RoBERTa base model with classification head
        - Dropout rate of 0.1 for regularization
        - Maximum sequence length of 256 tokens

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
    doc_classification_model = RobertaDocClassificationModel(n_doc_labels=10)
    optimizer = torch.optim.AdamW(doc_classification_model.parameters(), lr=3e-5)
    criterion = nn.CrossEntropyLoss()

    # Load and prepare datasets
    train_df = pd.read_csv("data/processed/train_data.csv")
    val_df = pd.read_csv("data/processed/val_data.csv")

    # Create dataset objects
    train_dataset = TextClassificationDataset(
        train_df["processed_text"], train_df["label"]
    )
    val_dataset = TextClassificationDataset(val_df["processed_text"], val_df["label"])

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Execute training
    train_doc_classification_model(
        doc_classification_model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        n_epochs=40,
        model_path="models/roberta_classification/",
    )


if __name__ == "__main__":
    train()
