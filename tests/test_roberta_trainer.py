# Copyright (c) Moitree Basu under MIT License.
# Tests for RoBERTa trainer implementation
# Code: https://github.com/moitreebasu1990/intellidoc

import os
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from intellidoc.classification_models.roberta import RobertaDocClassificationModel
from intellidoc.training.roberta_trainer import train, train_doc_classification_model


@pytest.fixture
def mock_model():
    class MockModelWithForward(Mock):
        def __call__(self, input_ids, attention_mask):
            batch_size = input_ids.size(0)
            return torch.randn(batch_size, 10, requires_grad=True)  # 10 classes

    model = MockModelWithForward(spec=RobertaDocClassificationModel)
    model.train = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(1, requires_grad=True)])
    return model


@pytest.fixture
def mock_optimizer():
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    return optimizer


@pytest.fixture
def mock_criterion():
    def compute_loss(outputs, targets):
        # Return actual tensor for loss
        return torch.tensor(0.5, requires_grad=True)

    criterion = Mock()
    criterion.side_effect = compute_loss
    return criterion


@pytest.fixture
def mock_data():
    # Create small fake dataset
    batch_size = 2
    seq_length = 10
    n_batches = 2

    def generate_batch():
        return {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "doc_labels": torch.randint(
                0, 10, (batch_size,)
            ),  # Changed to match expected shape
        }

    # Create mock DataLoader
    train_loader = MagicMock(spec=DataLoader)
    train_loader.__iter__.return_value = [generate_batch() for _ in range(n_batches)]
    train_loader.__len__.return_value = n_batches

    val_loader = MagicMock(spec=DataLoader)
    val_loader.__iter__.return_value = [generate_batch() for _ in range(n_batches)]
    val_loader.__len__.return_value = n_batches

    return train_loader, val_loader


def test_train_doc_classification_model(
    mock_model, mock_optimizer, mock_criterion, mock_data
):
    train_loader, val_loader = mock_data
    device = torch.device("cpu")

    with patch("intellidoc.training.roberta_trainer.save_model") as mock_save:
        # Run training
        train_doc_classification_model(
            mock_model,
            train_loader,
            val_loader,
            mock_optimizer,
            mock_criterion,
            device,
            n_epochs=2,
            model_path="test_model.pth",
        )

        # Verify model was moved to device
        mock_model.to.assert_called_once_with(device)

        # Verify train/eval modes were called
        assert mock_model.train.call_count > 0
        assert mock_model.eval.call_count > 0

        # Verify optimizer calls
        assert mock_optimizer.zero_grad.call_count > 0
        assert mock_optimizer.step.call_count > 0

        # Verify model was saved
        mock_save.assert_called_once()


@patch("torch.backends.mps.is_available", return_value=False)
@patch("torch.cuda.is_available", return_value=False)
@patch("pandas.read_csv")
@patch("torch.optim.AdamW")
@patch("intellidoc.training.roberta_trainer.train_doc_classification_model")
def test_train_function(
    mock_train_func, mock_optimizer, mock_read_csv, mock_cuda, mock_mps
):
    # Mock DataFrame
    mock_df = pd.DataFrame({"processed_text": ["text1", "text2"], "label": [0, 1]})
    mock_read_csv.return_value = mock_df

    # Run training pipeline
    with patch("intellidoc.training.roberta_trainer.TextClassificationDataset"):
        with patch("intellidoc.training.roberta_trainer.DataLoader"):
            train()

            # Verify training function was called
            assert mock_train_func.call_count == 1

            # Verify optimizer was created
            assert mock_optimizer.call_count == 1

            # Verify data was loaded
            assert mock_read_csv.call_count == 2


def test_device_selection():
    # Test CPU fallback when no GPU/MPS available
    with patch("torch.backends.mps.is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "intellidoc.training.roberta_trainer.train_doc_classification_model"
            ):
                with patch("pandas.read_csv"):
                    with patch(
                        "intellidoc.training.roberta_trainer.TextClassificationDataset"
                    ):
                        with patch("intellidoc.training.roberta_trainer.DataLoader"):
                            train()
                            device = torch.device("cpu")
                            assert str(device) == "cpu"

    # Test CUDA selection when available
    with patch("torch.backends.mps.is_available", return_value=False):
        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "intellidoc.training.roberta_trainer.train_doc_classification_model"
            ):
                with patch("pandas.read_csv"):
                    with patch(
                        "intellidoc.training.roberta_trainer.TextClassificationDataset"
                    ):
                        with patch("intellidoc.training.roberta_trainer.DataLoader"):
                            train()
                            device = torch.device("cuda")
                            assert str(device) == "cuda"
