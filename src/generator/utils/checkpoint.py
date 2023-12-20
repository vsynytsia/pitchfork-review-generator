import logging
from pathlib import Path

import torch
from torch import nn
import pickle
from src.config import ModelConfig
from src.generator.utils.dataset import Vocabulary

logger = logging.getLogger(__name__)


def get_checkpoint(model: nn.Module, epoch: int, vocab: Vocabulary, **kwargs) -> dict:
    """
    Creates a dictionary representing the state of the model and given keyword arguments.

    Parameters:
    - model (nn.Module): The model to be processed
    - epoch (int): The current epoch number
    - vocab (Vocabulary): Vocabulary used for training
    - kwargs: Additional parameters to include in the state dictionary. If a key is
            in ["optimizer", "scheduler"], calls its state_dict() method

    Returns:
    - The state dictionary with keys being argument names and values their corresponding states
    """

    state_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "vocab": vocab
    }

    keys_with_state_dict = ["optimizer", "scheduler"]

    for key, value in kwargs.items():
        if key in keys_with_state_dict:
            state_dict[key] = value.state_dict()
        else:
            state_dict[key] = value

    return state_dict


def save_checkpoint(checkpoint_folder: str,
                    model: nn.Module,
                    vocab: Vocabulary,
                    epoch: int,
                    **kwargs):
    """
    Saves a checkpoint of the model at a given epoch.

    Parameters:
    - model (nn.Module): The model to be processed.
    - epoch (int): The current epoch number.
    - vocab (Vocabulary): Vocabulary used for training
    - kwargs: Additional parameters to include in the state dictionary. If a key is
            in ["optimizer", "scheduler"], calls its state_dict() method.
    """

    checkpoint_folder = Path(checkpoint_folder)
    if not checkpoint_folder.exists():
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating {checkpoint_folder.as_posix()} folder")

    vocab_path = Path(checkpoint_folder) / "vocab.pkl"
    if not vocab_path.exists():
        logger.info(f"Creating {vocab_path.as_posix()}")

        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)

    state_dict = get_checkpoint(model, epoch, vocab, **kwargs)
    checkpoint_path = Path(checkpoint_folder) / f"checkpoint-{epoch}.pt"

    torch.save(state_dict, checkpoint_path)
    logging.info(f"Successfully saved checkpoint {checkpoint_path}")


def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load a checkpoint of the model at a given epoch

    Parameters:
    - checkpoint_path (str): The path to the checkpoint

    Returns:
    - The loaded state dictionary with keys being argument names and values their corresponding states
    """

    return torch.load(checkpoint_path, map_location=ModelConfig.DEVICE)
