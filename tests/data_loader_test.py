# Copyright (c) Moitree Basu under MIT License.
# Tests for data loader implementation
# Code: https://github.com/moitreebasu1990/intellidoc

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from intellidoc.data_processing.data_loader import partitioning_data


def test_partitioning_data(tmpdir="tests/temp_data/"):
    data = {
        "text": [
            "sample text 1",
            "sample text 2",
            "sample text 3",
            "sample text 4",
            "sample text 5",
            "sample text 6",
            "sample text 7",
            "sample text 8",
            "sample text 9",
            "sample text 10",
            "sample text 11",
            "sample text 12",
            "sample text 13",
            "sample text 14",
            "sample text 15",
            "sample text 16",
            "sample text 17",
            "sample text 18",
            "sample text 19",
            "sample text 20",
        ],
        "label": [
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
            "A",
            "B",
        ],
    }
    df = pd.DataFrame(data)

    # Save the DataFrame to a temporary CSV file
    input_csv = tmpdir + "input.csv"
    df.to_csv(input_csv, index=False)

    # Define output CSV paths
    output_csv_train = tmpdir + "train.csv"
    output_csv_val = tmpdir + "val.csv"
    output_csv_maintain = tmpdir + "maintain.csv"

    # Call the function
    partitioning_data(
        str(input_csv),
        str(output_csv_train),
        str(output_csv_val),
        str(output_csv_maintain),
    )

    # Load the output CSV files
    train_df = pd.read_csv(output_csv_train)
    val_df = pd.read_csv(output_csv_val)
    maintain_df = pd.read_csv(output_csv_maintain)

    # Check that the splits are approximately correct
    assert len(train_df) == 14
    assert len(val_df) == 4
    assert len(maintain_df) == 2

    # Check that the label encoding is consistent
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    assert set(train_df["label"]).issubset(set(df["label"]))
    assert set(val_df["label"]).issubset(set(df["label"]))
    assert set(maintain_df["label"]).issubset(set(df["label"]))
