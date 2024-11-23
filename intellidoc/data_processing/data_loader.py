# Copyright (c) Moitree Basu under MIT License.
# Data loading utilities for document classification
# Code: https://github.com/moitreebasu1990/intellidoc

"""Data loader module for partitioning dataset into train, validation and maintenance sets.

This module provides functionality to split a dataset into training, validation, and
maintenance sets while preserving the label distribution through stratification.
The splits are done in two steps:
1. 90:10 split for training+validation vs maintenance
2. 80:20 split of the 90% portion into training vs validation

The input CSV is expected to have the following columns:
    - text: The raw text data
    - label: The target labels
    - tokens: Tokenized text
    - filtered_tokens: Cleaned and filtered tokens
    - lemmatized_tokens: Lemmatized form of tokens
    - pos_tags: Part-of-speech tags
    - entities: Named entities
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def partitioning_data(input_csv, output_csv_train, output_csv_val, output_csv_maintain):
    """Partitions input data into training, validation and maintenance sets.

    Args:
        input_csv (str): Path to the input CSV file containing the full dataset.
        output_csv_train (str): Path where the training dataset will be saved.
        output_csv_val (str): Path where the validation dataset will be saved.
        output_csv_maintain (str): Path where the maintenance dataset will be saved.

    The function performs the following steps:
        1. Loads the data from input CSV
        2. Encodes labels to integers
        3. Splits data into 90% training+validation and 10% maintenance
        4. Further splits the 90% into 80% training and 20% validation
        5. Saves all three splits to separate CSV files
    """
    # Load the data into a DataFrame
    df = pd.read_csv(input_csv)

    # Convert labels to integers for stratification
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # Save the label mapping to a CSV file
    label_mapping = pd.DataFrame(
        {
            "original_label": label_encoder.classes_,
            "encoded_label": range(len(label_encoder.classes_)),
        }
    )
    label_mapping.to_csv("./data/processed/label_mapping.csv", index=False)

    # Split the data into training and maintenance sets (90:10)
    train_df, maintain_df = train_test_split(
        df, stratify=df["label"], test_size=0.1, random_state=42
    )

    # Further split the training data into training and validation sets (80:20)
    train_df, val_df = train_test_split(
        train_df, stratify=train_df["label"], test_size=0.2, random_state=42
    )

    # Save the partitioned datasets
    train_df.to_csv(output_csv_train, index=False)
    val_df.to_csv(output_csv_val, index=False)
    maintain_df.to_csv(output_csv_maintain, index=False)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Partition dataset into training, validation, and maintenance sets"
    )

    # Define command line arguments with default paths
    parser.add_argument(
        "--input_csv",
        type=str,
        default="./data/processed/final_data.csv",
        help="Path to input CSV file containing the full dataset",
    )
    parser.add_argument(
        "--output_csv_train",
        type=str,
        default="./data/processed/train_data.csv",
        help="Path to save the training dataset (72% of full data)",
    )
    parser.add_argument(
        "--output_csv_val",
        type=str,
        default="./data/processed/val_data.csv",
        help="Path to save the validation dataset (18% of full data)",
    )
    parser.add_argument(
        "--output_csv_maintain",
        type=str,
        default="./data/processed/maintain_data.csv",
        help="Path to save the maintenance dataset (10% of full data)",
    )

    # Parse arguments and execute data partitioning
    args = parser.parse_args()
    partitioning_data(
        args.input_csv,
        args.output_csv_train,
        args.output_csv_val,
        args.output_csv_maintain,
    )
