# Copyright (c) Moitree Basu under MIT License.
# Tests for RoBERTa model implementation
# Code: https://github.com/moitreebasu1990/intellidoc

import pytest
import torch

from intellidoc.classification_models.roberta import RobertaDocClassificationModel


@pytest.fixture
def model_params():
    return {
        "n_doc_labels": 3,
        "p_dropout": 0.2,
        "dim_hidden": 768,
        "pretrained_model_name": "roberta-base",
    }


@pytest.fixture
def sample_batch():
    batch_size = 2
    seq_length = 10
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(batch_size, seq_length),
    }


def test_model_initialization(model_params):
    model = RobertaDocClassificationModel(**model_params)

    # Check if all components are present
    assert hasattr(model, "model"), "RoBERTa base model not initialized"
    assert hasattr(model, "dropout"), "Dropout layer not initialized"
    assert hasattr(model, "classifier"), "Classifier not initialized"

    # Check model configuration
    assert model.pretrained_model_name == model_params["pretrained_model_name"]
    assert model.dropout.p == model_params["p_dropout"]
    assert model.classifier.out_features == model_params["n_doc_labels"]
    assert model.classifier.in_features == model_params["dim_hidden"]


def test_model_forward(model_params, sample_batch):
    model = RobertaDocClassificationModel(**model_params)

    # Run forward pass
    outputs = model(**sample_batch)

    # Check output shape
    batch_size = sample_batch["input_ids"].shape[0]
    assert outputs.shape == (
        batch_size,
        model_params["n_doc_labels"],
    ), f"Expected output shape {(batch_size, model_params['n_doc_labels'])}, got {outputs.shape}"

    # Check output type
    assert isinstance(outputs, torch.Tensor), "Output should be a torch.Tensor"
    assert outputs.dtype == torch.float32, "Output should be float32"


def test_model_dropout(model_params, sample_batch):
    # Set high dropout for testing
    model_params["p_dropout"] = 0.9
    model = RobertaDocClassificationModel(**model_params)

    # Set to training mode
    model.train()

    # Get outputs with dropout
    out1 = model(**sample_batch)
    out2 = model(**sample_batch)

    # Outputs should be different in training mode due to dropout
    assert not torch.allclose(
        out1, out2
    ), "Dropout doesn't seem to be working in training mode"

    # Set to evaluation mode
    model.eval()

    # Get outputs without dropout
    with torch.no_grad():
        out1 = model(**sample_batch)
        out2 = model(**sample_batch)

    # Outputs should be identical in eval mode
    assert torch.allclose(out1, out2), "Outputs should be identical in eval mode"


def test_get_config(model_params):
    model = RobertaDocClassificationModel(**model_params)
    config = model.get_config()

    # Check if all expected keys are present
    expected_keys = ["n_doc_labels", "p_dropout", "dim_hidden", "pretrained_model_name"]
    assert all(
        key in config for key in expected_keys
    ), f"Missing keys in config. Expected {expected_keys}, got {list(config.keys())}"

    # Check if values match initialization parameters
    assert config["n_doc_labels"] == model_params["n_doc_labels"]
    assert config["p_dropout"] == model_params["p_dropout"]
    assert config["dim_hidden"] == model_params["dim_hidden"]
    assert config["pretrained_model_name"] == model_params["pretrained_model_name"]


def test_invalid_num_labels():
    with pytest.raises(ValueError, match="n_doc_labels must be positive"):
        RobertaDocClassificationModel(n_doc_labels=0)


def test_invalid_dropout():
    with pytest.raises(ValueError, match="p_dropout must be between 0 and 1"):
        RobertaDocClassificationModel(n_doc_labels=2, p_dropout=1.5)


def test_forward_input_validation(model_params):
    model = RobertaDocClassificationModel(**model_params)

    # Test with mismatched batch sizes
    batch_size = 2
    seq_length = 10
    invalid_batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(
            batch_size + 1, seq_length
        ),  # Different batch size
    }

    with pytest.raises(RuntimeError):
        model(**invalid_batch)

    # Test with mismatched sequence lengths
    invalid_batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_length)),
        "attention_mask": torch.ones(
            batch_size, seq_length + 1
        ),  # Different sequence length
    }

    with pytest.raises(RuntimeError):
        model(**invalid_batch)
