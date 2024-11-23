# Copyright (c) Moitree Basu under MIT License.
# Dataset classes for document classification
# Code: https://github.com/moitreebasu1990/intellidoc

"""Dataset module for text classification using RoBERTa model.

This module provides a PyTorch Dataset implementation for text classification tasks
using the RoBERTa tokenizer. It handles both single string inputs and lists of tokens,
converting them into the appropriate format for the RoBERTa model.
"""

from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


class TextClassificationDataset(Dataset):
    """A PyTorch Dataset for text classification tasks using RoBERTa.

    This dataset class handles the preprocessing of text data for RoBERTa-based
    classification models. It tokenizes the input text using RoBERTa's tokenizer
    and creates attention masks for proper model input.

    Attributes:
        input (Union[List[str], List[List[str]]]): Input texts or tokens.
        doc_labels (List[int], optional): Classification labels for each document.
        max_length (int): Maximum sequence length for tokenization.
        tokenizer (RobertaTokenizer): Pre-trained RoBERTa tokenizer.
    """

    def __init__(
        self,
        input: Union[List[str], List[List[str]]],
        doc_labels: List[int] = None,
        max_length: int = 256,
    ):
        super().__init__()
        self.input = input
        self.doc_labels = doc_labels
        self.max_length = max_length
        # Load the RoBERTa tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.input)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.

        This method processes the text at the given index by:
        1. Joining tokens if the input is pre-tokenized
        2. Encoding the text using RoBERTa tokenizer
        3. Creating attention masks
        4. Packaging everything into a dictionary

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - input_ids: Encoded token IDs (shape: max_length)
                - attention_mask: Attention mask (shape: max_length)
                - doc_labels: Document label if available (shape: 1)
        """
        # Join tokens if input is pre-tokenized, otherwise use raw text
        text = " ".join(self.input[index])

        # Tokenize and encode the text with padding and truncation
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Extract and squeeze tensors to remove batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create the sample dictionary with all necessary tensors
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "doc_labels": self.doc_labels[index],
        }

        return item
