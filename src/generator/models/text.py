import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from src.config import ModelConfig
from src.generator.utils.batches import get_batches, one_hot_encode
from src.generator.utils.checkpoint import save_checkpoint, load_checkpoint
from src.generator.utils.dataset import Vocabulary
from src.generator.utils.preprocess import clean_string
from .lstm import CharacterLevelLSTM

logger = logging.getLogger(__name__)


class TextGenerationModel:
    """Text generation model which utilizes CharLevel LSTM for training and inference."""

    def __init__(self,
                 lstm: CharacterLevelLSTM,
                 vocab: Vocabulary,
                 train_dataset: Optional[np.ndarray] = None,
                 test_dataset: Optional[np.ndarray] = None,
                 n_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 seq_length: Optional[int] = None,
                 optimizer: Optional[Optimizer] = None,
                 scheduler: Optional[lr_scheduler] = None,
                 criterion: Optional[nn.Module] = None,
                 checkpoint_folder: Optional[str] = None,
                 checkpoint_every: Optional[int] = None):
        """
        Initialize the text generation model

        Parameters:
        - lstm (CharacterLevelLSTM): The LSTM model
        - vocab (Vocabulary): The vocabulary used for training and inference
        - train_dataset (np.ndarray): Dataset used for training
        - test_dataset (np.ndarray): Dataset used for validation
        - n_epochs (int): The number of epochs for training
        - batch_size (int): The batch size for training
        - seq_length (int): The sequence length for training
        - optimizer (Optimizer): The optimizer for training
        - scheduler (lr_scheduler): The learning rate scheduler for training
        - criterion (nn.Module): The loss function
        - checkpoint_folder (str): Path to folder where checkpoints will be saved
        - checkpoint_every (int): Number of epochs passed between checkpoint. Defaults to 10
        """

        super().__init__()

        self.lstm = lstm
        self.vocab = vocab
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_every = checkpoint_every

        self.train_losses, self.val_losses = np.array([]), np.array([])

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, vocab: Vocabulary) -> 'TextGenerationModel':
        lstm = CharacterLevelLSTM(
            n_hidden=ModelConfig.N_HIDDEN,
            n_layers=ModelConfig.N_LAYERS,
            dropout_prob=ModelConfig.DROPOUT_PROB,
            vocab_size=vocab.size
        ).to(ModelConfig.DEVICE)
        checkpoint = load_checkpoint(checkpoint_path)
        lstm.load_state_dict(checkpoint["model_state_dict"])
        return cls(lstm=lstm, vocab=vocab)

    def train_one_epoch(self):
        """
        Train the LSTM model for one epoch.

        Returns:
        - Numpy array of loss history over batches.
        """

        self.lstm.train()
        hidden = self.lstm.init_hidden_state(self.batch_size)
        loss_history = np.array([])

        for inputs_batch, targets_batch in get_batches(self.train_dataset, self.batch_size, self.seq_length):
            inputs_batch = one_hot_encode(inputs_batch, self.vocab.size)
            inputs_batch, targets_batch = inputs_batch.to(ModelConfig.DEVICE), targets_batch.to(ModelConfig.DEVICE)

            hidden = tuple([h.data for h in hidden])

            self.lstm.zero_grad()
            out, last_hidden_state = self.lstm(inputs_batch, hidden)
            loss = self.criterion(out, targets_batch.view(self.batch_size * self.seq_length).long())
            loss.backward()

            nn.utils.clip_grad_norm_(self.lstm.parameters(), 1)
            self.optimizer.step()

            loss_history = np.append(loss_history, loss.cpu().detach().numpy())
        return loss_history

    def validate_one_epoch(self):
        """
        Validate the LSTM model for one epoch.

        Returns:
        - Numpy array of loss history over batches.
        """

        self.lstm.eval()
        hidden = self.lstm.init_hidden_state(self.batch_size)

        loss_history = np.array([])

        for inputs_batch, targets_batch in get_batches(self.test_dataset, self.batch_size, self.seq_length):
            inputs_batch = one_hot_encode(inputs_batch, self.vocab.size)
            inputs_batch, targets_batch = inputs_batch.to(ModelConfig.DEVICE), targets_batch.to(ModelConfig.DEVICE)

            hidden = tuple([h.data for h in hidden])

            with torch.no_grad():
                out, last_hidden_state = self.lstm(inputs_batch, hidden)
                loss = self.criterion(out, targets_batch.view(self.batch_size * self.seq_length).long())

            loss_history = np.append(loss_history, loss.cpu().detach().numpy())
        return loss_history

    def predict_next_token(self, hidden, current_token: str, top_k: int):
        """
        Predict the next token given the current one and the hidden state.

        Parameters:
        - hidden (torch.Tensor): The hidden state tensor.
        - current_token (str): The current token.
        - top_k (int): The top K predictions to consider.

        Returns:
        - Tuple of next token and hidden state tensor.
        """

        self.lstm.eval()

        hidden = tuple([h.data for h in hidden])
        inputs = torch.tensor([[self.vocab.char2idx[current_token]]]).long()
        inputs = one_hot_encode(inputs, self.vocab.size).to(ModelConfig.DEVICE)

        with torch.no_grad():
            output, hidden = self.lstm(inputs, hidden)

            topk_values, topk_indices = torch.topk(output.squeeze(0).cpu(), k=top_k)
            topk_values, topk_indices = topk_values.numpy(), topk_indices.numpy()

            output[output < topk_values[-1]] = float('-inf')
            probs = F.softmax(output, dim=-1).data
            sample = torch.multinomial(probs, 1)
            return self.vocab.idx2char[sample.item()], hidden

    def generate_text(self, text_length: int, prime: Optional[str] = None, top_k: int = 5) -> str:
        """
        Generate text using the trained LSTM model.

        Parameters:
        - text_length (int): Length of text to generate.
        - prime (str): Initial seed string for generating the text.
        - top_k (int): Top K predictions to consider.

        Returns:
        - Generated text
        """

        self.lstm.eval()

        generated_text = list(clean_string(prime))
        hidden = self.lstm.init_hidden_state(batch_size=1)

        for token in generated_text[:-1]:
            _, hidden = self.predict_next_token(hidden, token, top_k)

        for _ in range(text_length - len(generated_text)):
            next_token, hidden = self.predict_next_token(hidden, generated_text[-1], top_k)
            generated_text.append(next_token)

        return "".join(generated_text)

    def train(self):
        """Train the LSTM model for a specified number of epochs."""

        for i in range(self.n_epochs):
            train_time_start = time.time()
            mean_train_loss = self.train_one_epoch().mean()
            train_time_end = time.time()

            val_time_start = time.time()
            mean_val_loss = self.validate_one_epoch().mean()
            val_time_end = time.time()

            self.scheduler.step(mean_val_loss)

            self.train_losses = np.append(self.train_losses, mean_train_loss)
            self.val_losses = np.append(self.val_losses, mean_val_loss)

            logger.info(f"Epoch #{i}"
                        f"\tTrain loss = {mean_train_loss:.2f}, time taken: {(train_time_end - train_time_start):.2f}s"
                        f"\tVal loss = {mean_val_loss:.2f}, time taken: {(val_time_end - val_time_start):.2f}s"
                        f"\nExample of generated text: {self.generate_text(prime='This album is', text_length=100)}")

            if i > 0 and i % self.checkpoint_every == 0:
                save_checkpoint(checkpoint_folder=self.checkpoint_folder,
                                epoch=i,
                                model=self.lstm,
                                vocab=self.vocab,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                train_loss=self.train_losses[i],
                                val_loss=self.val_losses[i])
