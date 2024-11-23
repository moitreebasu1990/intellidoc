# Copyright (c) Moitree Basu under MIT License.
# Model saving and loading utilities
# Code: https://github.com/moitreebasu1990/intellidoc

from pathlib import Path
from typing import Union

import torch

from intellidoc.classification_models.roberta import RobertaDocClassificationModel


def save_model(
    model: "RobertaDocClassificationModel",
    save_dir: Union[str, Path],
    filename: str = "model.pt",
) -> Path:
    """
    Save the RoBERTa classification model.

    Args:
        model: The model to save
        save_dir: Directory to save the model
        filename: Name of the model checkpoint file

    Returns:
        Path: Path to the saved model checkpoint
    """
    # Convert to Path object and create directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save RoBERTa base model
    roberta_path = save_dir / "roberta"
    model.model.save_pretrained(roberta_path)

    # Save custom layers and configuration
    model_state = {
        "classifier_state": model.classifier.state_dict(),
        "dropout_state": model.dropout.state_dict(),
        "config": model.get_config(),
    }

    # Save checkpoint
    checkpoint_path = save_dir / filename
    torch.save(model_state, checkpoint_path)

    return checkpoint_path


def load_model(
    load_dir: Union[str, Path], filename: str = "model.pt"
) -> RobertaDocClassificationModel:
    """
    Load the RoBERTa classification model.

    Args:
        load_dir: Directory containing the saved model
        filename: Name of the model checkpoint file

    Returns:
        RobertaDocClassificationModel: Loaded model
    """
    # Convert to Path object
    load_dir = Path(load_dir)

    # Load checkpoint
    checkpoint_path = load_dir / filename
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get saved configuration
    config = checkpoint["config"]

    # Use saved RoBERTa weights
    roberta_path = str(load_dir / "roberta")
    model = RobertaDocClassificationModel(
        n_doc_labels=config["n_doc_labels"],
        p_dropout=config["p_dropout"],
        dim_hidden=config["dim_hidden"],
        pretrained_model_name=roberta_path,
    )

    # Load custom layers
    model.classifier.load_state_dict(checkpoint["classifier_state"])
    model.dropout.load_state_dict(checkpoint["dropout_state"])

    # model.eval()
    return model
