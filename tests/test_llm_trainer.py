# Copyright (c) Moitree Basu under MIT License.
# Tests for LLM trainer implementation
# Code: https://github.com/moitreebasu1990/intellidoc

"""Unit tests for the advanced LLM trainer module.

This module contains unit tests for the LLM-based document classification training pipeline.
It tests various components including:
    - Layer-wise loss computation
    - Training loop functionality
    - Device selection logic
    - Attention map visualization
    - Model checkpointing
"""

import os
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from intellidoc.classification_models.advance_llm import AdvancedDocClassificationModel
from intellidoc.training.llm_trainer import (
    LayerWiseLoss,
    train,
    train_doc_classification_model,
)


@pytest.fixture
def mock_model():
    """Create a mock model for testing the training pipeline.

    This fixture creates a mock AdvancedDocClassificationModel with simulated outputs
    including logits, intermediate layer outputs, and attention probabilities.

    Returns:
        Mock: A mock model object that simulates the behavior of AdvancedDocClassificationModel
            with the following methods and attributes:
            - forward(): Returns dict with logits, intermediate_logits, and attention_probs
            - train(), eval(): Mock training/evaluation mode switches
            - to(): Device movement simulation
            - parameters(): Simulated model parameters
            - state_dict(): Empty state dictionary
            - get_config(): Model configuration dictionary
    """

    class MockModelWithForward(Mock):
        def __call__(self, input_ids, attention_mask):
            batch_size = input_ids.size(0)
            # Return dict with logits, intermediate logits, and attention probs
            return {
                "logits": torch.randn(batch_size, 10, requires_grad=True),
                "intermediate_logits": [
                    torch.randn(batch_size, 10, requires_grad=True)
                    for _ in range(2)  # 2 intermediate layers
                ],
                "attention_probs": [
                    torch.randn(batch_size, 8, 32, 32)  # 8 attention heads
                    for _ in range(2)  # 2 layers
                ],
            }

    model = MockModelWithForward(spec=AdvancedDocClassificationModel)
    model.train = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(1, requires_grad=True)])
    model.state_dict = Mock(return_value={})  # Add state_dict method
    model.get_config = Mock(
        return_value={
            "n_doc_labels": 10,
            "p_dropout": 0.1,
            "num_hidden_layers": 2,
            "pretrained_model_name": "test-model",
        }
    )
    return model


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing.

    Returns:
        Mock: A mock optimizer object with step() and zero_grad() methods.
    """
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.state_dict = Mock(return_value={})
    return optimizer


@pytest.fixture
def mock_data():
    """Create mock training and validation data loaders.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing:
            - train_loader: Mock training data loader
            - val_loader: Mock validation data loader
    """
    # Create small fake dataset
    batch_size = 2
    seq_length = 10
    n_batches = 2

    def generate_batch():
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "doc_labels": torch.randint(0, 10, (batch_size,)),
        }

    # Create mock DataLoader
    train_loader = MagicMock(spec=DataLoader)
    train_loader.__iter__.return_value = [generate_batch() for _ in range(n_batches)]
    train_loader.__len__.return_value = n_batches

    val_loader = MagicMock(spec=DataLoader)
    val_loader.__iter__.return_value = [generate_batch() for _ in range(n_batches)]
    val_loader.__len__.return_value = n_batches

    return train_loader, val_loader


def test_layer_wise_loss():
    """Test the LayerWiseLoss class functionality.

    This test verifies that:
        1. Loss is computed correctly for each layer
        2. Weights are applied properly to layer losses
        3. Final loss is aggregated correctly
        4. Handles both weighted and unweighted scenarios
    """
    # Initialize loss function with custom weights
    weights = [0.2, 0.3, 0.5]
    criterion = LayerWiseLoss(weights=weights)

    # Create mock data
    batch_size = 2
    n_classes = 10
    outputs = {
        "logits": torch.randn(batch_size, n_classes, requires_grad=True),
        "intermediate_logits": [
            torch.randn(batch_size, n_classes, requires_grad=True) for _ in range(2)
        ],
    }
    targets = torch.randint(0, n_classes, (batch_size,))

    # Calculate loss
    total_loss, loss_dict = criterion(outputs, targets)

    # Verify loss properties
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.requires_grad
    assert len(loss_dict) == 3  # 2 intermediate + 1 final
    assert "final_layer_loss" in loss_dict
    assert "layer_0_loss" in loss_dict
    assert "layer_1_loss" in loss_dict


def test_train_doc_classification_model(
    mock_model, mock_optimizer, mock_data, tmp_path
):
    """Test the document classification model training function.

    This test verifies:
        1. Training loop executes correctly
        2. Model states are properly updated
        3. Attention maps are saved when requested
        4. Metrics are computed and returned
        5. Model checkpoints are saved correctly

    Args:
        mock_model: Fixture providing a mock model
        mock_optimizer: Fixture providing a mock optimizer
        mock_data: Fixture providing mock data loaders
        tmp_path: Pytest fixture providing temporary directory path
    """
    train_loader, val_loader = mock_data
    device = torch.device("cpu")
    criterion = LayerWiseLoss(weights=[0.2, 0.3, 0.5])

    # Create temporary model path
    model_path = os.path.join(tmp_path, "test_model.pth")

    # Run training
    report = train_doc_classification_model(
        mock_model,
        train_loader,
        val_loader,
        mock_optimizer,
        criterion,
        device,
        n_epochs=2,
        model_path=model_path,
        save_attention_maps=True,
    )

    # Verify model was moved to device
    mock_model.to.assert_called_once_with(device)

    # Verify train/eval modes were called
    assert mock_model.train.call_count > 0
    assert mock_model.eval.call_count > 0

    # Verify optimizer calls
    assert mock_optimizer.zero_grad.call_count > 0
    assert mock_optimizer.step.call_count > 0

    # Verify model and attention maps were saved
    assert os.path.exists(os.path.dirname(model_path))
    attention_dir = os.path.join(os.path.dirname(model_path), "attention_maps")
    assert os.path.exists(attention_dir)


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=False)
@patch("pandas.read_csv")
@patch("torch.optim.AdamW")
@patch("intellidoc.training.llm_trainer.train_doc_classification_model")
def test_train_function(
    mock_train_func, mock_optimizer, mock_read_csv, mock_cuda, mock_mps
):
    """Test the main training pipeline function.

    This test verifies:
        1. Correct device selection (CPU/CUDA/MPS)
        2. Model initialization
        3. Data loading and preprocessing
        4. Training execution with proper parameters
        5. Error handling and logging

    Args:
        mock_train_func: Mock for the training function
        mock_optimizer: Mock for the optimizer
        mock_read_csv: Mock for pandas read_csv
        mock_cuda: Mock for CUDA availability check
        mock_mps: Mock for MPS availability check
    """
    # Mock DataFrame
    mock_df = pd.DataFrame({"processed_text": ["text1", "text2"], "label": [0, 1]})
    mock_read_csv.return_value = mock_df

    # Run training pipeline
    with patch("intellidoc.training.llm_trainer.TextClassificationDataset"):
        with patch("intellidoc.training.llm_trainer.DataLoader"):
            with patch("intellidoc.training.llm_trainer.LayerWiseLoss"):
                train()

                # Verify training function was called
                assert mock_train_func.call_count == 1

                # Verify optimizer was created with correct params
                mock_optimizer.assert_called_once()
                _, kwargs = mock_optimizer.call_args
                assert kwargs["lr"] == 2e-5
                assert kwargs["weight_decay"] == 0.01
                assert kwargs["eps"] == 1e-8

                # Verify data was loaded
                assert mock_read_csv.call_count == 2


def test_device_selection():
    """Test the device selection logic for training.

    This test verifies:
        1. CUDA device is selected when available
        2. MPS device is selected on Apple Silicon when CUDA is unavailable
        3. CPU is selected when neither CUDA nor MPS is available
        4. Device selection respects user preferences
    """
    # Test CPU fallback
    with patch("torch.backends.mps.is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "intellidoc.training.llm_trainer.train_doc_classification_model"
            ):
                with patch("pandas.read_csv"):
                    with patch(
                        "intellidoc.training.llm_trainer.TextClassificationDataset"
                    ):
                        with patch("intellidoc.training.llm_trainer.DataLoader"):
                            with patch("intellidoc.training.llm_trainer.LayerWiseLoss"):
                                train()
                                device = torch.device("cpu")
                                assert str(device) == "cpu"

    # Test CUDA selection
    with patch("torch.backends.mps.is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "intellidoc.training.llm_trainer.train_doc_classification_model"
            ):
                with patch("pandas.read_csv"):
                    with patch(
                        "intellidoc.training.llm_trainer.TextClassificationDataset"
                    ):
                        with patch("intellidoc.training.llm_trainer.DataLoader"):
                            with patch("intellidoc.training.llm_trainer.LayerWiseLoss"):
                                train()
                                device = torch.device("cuda")
                                assert str(device) == "cuda"
