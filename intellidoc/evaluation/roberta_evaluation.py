# Copyright (c) Moitree Basu under MIT License.
# RoBERTa-based document classification evaluation
# Code: https://github.com/moitreebasu1990/intellidoc

import os

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom classes or functions defined elsewhere in the project
from intellidoc.classification_models.roberta import RobertaDocClassificationModel
from intellidoc.data_processing.dataset import TextClassificationDataset
from intellidoc.utilities.save_load_model import load_model


def eval_doc_classification_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    model_path: str = "model.pth",  # New parameter for model path
) -> str:
    """
    Module to train a document classification model for a certain number of epochs.

    Args:
        model (nn.Module): The PyTorch model to train.
        val_loader (DataLoader): The DataLoader containing the validation data.
        device (torch.device): The device to use for training.

    Returns:
        str: A classification report containing the precision, recall, f1-score, and support for each class.
    """

    # Load model from checkpoint if provided
    if model_path and os.path.exists(model_path + "model.pth"):
        model = load_model(model_path, "model.pth")
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}")
        return

    # Move the model to the device
    model = model.to(device)

    # Set the model to evaluation mode
    model.eval()

    all_predictions = []
    all_labels = []

    # Initialize counters for validation accuracy
    correct_val = 0
    total_val = 0

    # Disable gradient computation
    with torch.no_grad():

        # Iterate over the validation data
        for batch in tqdm(val_loader):

            # Move the batch data to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            doc_labels = batch["doc_labels"].to(device)

            # Forward pass
            model_outputs = model(input_ids, attention_mask)

            # Get the predicted labels
            prediction = torch.argmax(model_outputs, dim=-1)

            # Convert the predictions and labels to numpy arrays and extend the lists
            all_predictions.extend(prediction.cpu().numpy())
            all_labels.extend(doc_labels.squeeze(-1).cpu().numpy())

            # Calculate validation accuracy
            correct_val += (prediction == doc_labels.squeeze(-1)).sum().item()
            total_val += doc_labels.size(0)

    # Return a classification report
    return classification_report(all_labels, all_predictions)


def eval():
    # Checking for device availability
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Initializing the document classification model with n_doc_labels labels
    doc_classification_model = RobertaDocClassificationModel(n_doc_labels=10)

    # Read training and validation datasets
    val_df = pd.read_csv("data/processed/val_data.csv")

    # Initialize the training dataset
    val_dataset = TextClassificationDataset(val_df["processed_text"], val_df["label"])

    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Train a document classification model.
    doc_classification_report = eval_doc_classification_model(
        doc_classification_model,
        val_loader,
        device,
        model_path="models/roberta_classification/",
    )

    # Extract the F1 score from the classification report
    print(
        f"Doc classification model performance using {len(val_dataset)} validation examples."
    )
    print(doc_classification_report)
    doc_f1_score = float(doc_classification_report.split("\n")[-2].split()[-2])
    print(f"Doc classification model's F1-score: {doc_f1_score}")


if __name__ == "__main__":
    eval()
