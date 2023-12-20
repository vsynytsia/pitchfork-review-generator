from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler, Optimizer

from src.config import ModelConfig
from src.generator.utils.batches import get_batches, one_hot_encode


class CharacterLevelLSTM(nn.Module):
    """Character-level LSTM model"""

    def __init__(self,
                 n_hidden: int,
                 n_layers: int,
                 dropout_prob: float,
                 vocab_size: int):
        """
        Initialize LSTM model.

        Parameters:
        - n_hidden (int): The number of hidden units in the LSTM layer
        - n_layers (int): The number of LSTM layers
        - dropout_prob (float): The dropout probability for regularization
        - vocab_size (int): The size of the vocabulary or number of unique characters
        """

        super().__init__()

        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.dropout_prob = dropout_prob
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(input_size=vocab_size,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            dropout=dropout_prob,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(n_hidden, vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LSTM.

        Parameters:
        - x (torch.Tensor): The input tensor to the LSTM model
        - hidden (torch.Tensor): The hidden state tensor

        Returns:
        - Tuple of output and hidden state tensors
        """

        lstm_output, hidden = self.lstm(x, hidden)
        out = self.dropout(lstm_output)

        out = out.reshape(-1, self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state of LSTM with 0.

        Parameters:
        - batch_size (int): The batch size used for training

        Returns:
        - Tuple of hidden state tensors initialized with 0
        """

        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(ModelConfig.DEVICE),
                weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(ModelConfig.DEVICE))

