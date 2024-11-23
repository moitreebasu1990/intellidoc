# Copyright (c) Moitree Basu under MIT License.
# Tests for dataset implementation
# Code: https://github.com/moitreebasu1990/intellidoc

import pytest
import torch

from intellidoc.data_processing.dataset import TextClassificationDataset


@pytest.fixture
def sample_texts():
    return [
        "This is a test document",
        "Another test document",
        "Third test document for testing",
    ]


@pytest.fixture
def sample_labels():
    return [0, 1, 0]


def test_dataset_initialization(sample_texts, sample_labels):
    dataset = TextClassificationDataset(
        input=sample_texts, doc_labels=sample_labels, max_length=128
    )
    assert len(dataset) == len(sample_texts)
    assert dataset.max_length == 128
    assert dataset.doc_labels == sample_labels


def test_dataset_initialization_without_labels(sample_texts):
    dataset = TextClassificationDataset(input=sample_texts, max_length=128)
    assert len(dataset) == len(sample_texts)
    assert dataset.doc_labels is None


def test_dataset_getitem(sample_texts, sample_labels):
    dataset = TextClassificationDataset(
        input=sample_texts, doc_labels=sample_labels, max_length=128
    )

    item = dataset[0]
    assert isinstance(item, dict)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "doc_labels" in item

    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["doc_labels"], int)

    # Check shapes
    assert item["input_ids"].dim() == 1
    assert item["attention_mask"].dim() == 1


def test_dataset_with_token_list():
    # Test with list of tokens instead of strings
    tokens = [["this", "is", "test"], ["another", "test"]]
    dataset = TextClassificationDataset(input=tokens, doc_labels=[0, 1], max_length=128)
    assert len(dataset) == len(tokens)

    item = dataset[0]
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)


def test_dataset_max_length():
    texts = ["short text", "this is a much longer text that should be truncated"]
    dataset = TextClassificationDataset(input=texts, doc_labels=[0, 1], max_length=10)

    item = dataset[1]
    assert item["input_ids"].size(0) <= 10  # Account for special tokens
    assert item["attention_mask"].size(0) <= 10
