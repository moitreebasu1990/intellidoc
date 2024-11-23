# Copyright (c) Moitree Basu under MIT License.
# RoBERTa model architecture for document classification
# Code: https://github.com/moitreebasu1990/intellidoc

"""RoBERTa-based document classification model implementation.

This module implements a document classification model using the RoBERTa architecture
as the backbone. It adds a classification head on top of RoBERTa's pooled output
to perform document-level classification tasks.

The model architecture consists of:
    1. Pre-trained RoBERTa base model
    2. Dropout layer for regularization
    3. Linear classification head
"""

from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaDocClassificationModel(nn.Module):
    """Document classification model based on RoBERTa architecture.

    This class implements a neural network for document classification using RoBERTa
    as the backbone encoder. It adds a classification head on top of RoBERTa's
    pooled output for predicting document labels.

    Attributes:
        model (RobertaModel): Pre-trained RoBERTa model for feature extraction.
        dropout (nn.Dropout): Dropout layer for regularization.
        classifier (nn.Linear): Linear layer for classification.
        pretrained_model_name (str): Name of the pre-trained model used.
    """

    def __init__(
        self,
        n_doc_labels: int,
        p_dropout: float = 0.1,
        dim_hidden: int = 768,
        pretrained_model_name: str = "roberta-base",
    ):
        """Initialize the RoBERTa document classification model.

        Args:
            n_doc_labels (int): Number of target document classes.
            p_dropout (float, optional): Dropout probability for regularization.
                Defaults to 0.1.
            dim_hidden (int, optional): Dimension of the hidden representations.
                Defaults to 768 (RoBERTa base model's hidden size).
            pretrained_model_name (str, optional): Name of the pre-trained RoBERTa model.
                Defaults to "roberta-base".

        Raises:
            ValueError: If n_doc_labels is not positive or p_dropout is not between 0 and 1.
        """
        if n_doc_labels <= 0:
            raise ValueError("n_doc_labels must be positive")
        if not 0 <= p_dropout <= 1:
            raise ValueError("p_dropout must be between 0 and 1")

        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        # Load pre-trained RoBERTa model
        self.model = RobertaModel.from_pretrained(self.pretrained_model_name)
        # Add dropout for regularization
        self.dropout = nn.Dropout(p_dropout)
        # Add classification head
        self.classifier = nn.Linear(dim_hidden, n_doc_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Perform forward pass through the model.

        This method:
        1. Passes input through RoBERTa to get contextual representations
        2. Uses the pooled output for document-level representation
        3. Applies dropout for regularization
        4. Passes through the classification head to get logits

        Args:
            input_ids (torch.Tensor): Token IDs of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Attention mask of shape (batch_size, sequence_length).
                1 for tokens that are not masked, 0 for masked tokens.

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, n_doc_labels).

        Raises:
            RuntimeError: If input_ids and attention_mask have mismatched dimensions.
        """
        # Validate input dimensions
        if input_ids.shape != attention_mask.shape:
            raise RuntimeError(
                f"Mismatched dimensions: input_ids shape {input_ids.shape} != "
                f"attention_mask shape {attention_mask.shape}"
            )

        # Get RoBERTa's encoded representations
        roberta_output = self.model(input_ids, attention_mask)

        # Use pooled output for document-level classification
        sequence_output = roberta_output.pooler_output

        # Apply dropout for regularization
        sequence_output = self.dropout(sequence_output)

        # Get classification logits
        logits = self.classifier(sequence_output)

        return logits

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for saving and reproduction.

        Returns:
            Dict[str, Any]: Configuration dictionary containing:
                - n_doc_labels: Number of target classes
                - p_dropout: Dropout probability
                - dim_hidden: Hidden dimension size
                - pretrained_model_name: Name of pre-trained model used
        """
        return {
            "n_doc_labels": self.classifier.out_features,
            "p_dropout": self.dropout.p,
            "dim_hidden": self.classifier.in_features,
            "pretrained_model_name": self.pretrained_model_name,
        }
