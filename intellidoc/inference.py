"""Inference module for document classification using trained RoBERTa model.

This module provides functionality for real-time inference using a trained RoBERTa
document classification model. It includes:
    - Model loading and setup
    - Text preprocessing
    - Interactive prediction loop
    - Human-readable output formatting
"""

import pandas as pd
import torch
import re
import unicodedata
from html import unescape

from intellidoc.classification_models.roberta import RobertaDocClassificationModel
from intellidoc.data_processing.dataset import TextClassificationDataset
from intellidoc.utilities.save_load_model import load_model


def preprocess_text(text):
    """
    Preprocess the input text for document classification.
    
    Steps:
    1. HTML unescape
    2. Unicode normalization
    3. Convert to lowercase
    4. Remove special characters and numbers
    5. Remove extra whitespace
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # HTML unescape (convert HTML entities to characters)
    text = unescape(text)
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def load_inference_model(model_path="models/roberta_classification/", use_cpu=False):
    """
    Load the trained model for inference.
    """

    # Determine the device
    if not use_cpu:
        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device("cpu")

    # Load the model
    model = load_model(model_path, "model.pth")
    model.to(device)
    model.eval()

    return model, device


def predict_document_class(model, text, device):
    """
    Predict the class of the input text.
    """
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Create a temporary dataset for inference
    temp_df = pd.DataFrame({"processed_text": [processed_text], "label": [0]})
    inference_dataset = TextClassificationDataset(
        temp_df["processed_text"], temp_df["label"]
    )

    # Prepare the input
    with torch.no_grad():
        batch = inference_dataset[0]
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(device)

        # Get model prediction
        model_output = model(input_ids, attention_mask)
        prediction = torch.argmax(model_output, dim=-1)

        # Define class names (adjust according to your actual labels)
        class_names = [
            "business",
            "entertainment",
            "food",
            "graphics",
            "historical",
            "medical",
            "politics",
            "space",
            "sport",
            "technologie",
        ]

        return class_names[prediction.item()]


def main():
    # Load the model
    model, device = load_inference_model(use_cpu=True)

    print("Document Classification Inference")
    print("Type 'QUIT' to exit.")

    while True:
        # Get user input
        user_text = input("\nEnter a document text: ")

        # Check for quit condition
        if user_text.upper() == "QUIT":
            print("Exiting inference...")
            break

        # Predict and print the class
        predicted_class = predict_document_class(model, user_text, device)
        print(f"Predicted Class: {predicted_class}")


if __name__ == "__main__":
    main()
