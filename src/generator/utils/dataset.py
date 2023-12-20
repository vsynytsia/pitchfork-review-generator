from typing import Callable, Optional

import numpy as np
import pandas as pd


class Vocabulary:
    """A vocabulary which maintains mappings from char to idx and idx to char."""

    def __init__(self, data: list[str]):
        """
        Initializes a Vocabulary instance

        Parameters:
        - data (list[str]): A list of string data
        """

        self.unique_chars = set(set(" ".join(data)))
        self.size = len(self.unique_chars)

        self.idx2char = dict(enumerate(self.unique_chars))
        self.char2idx = {char: idx for idx, char in self.idx2char.items()}


class Dataset:
    """A dataset which manages train and test data along with the necessary vocabulary."""

    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        """
        Initializes a Dataset instance.

        Parameters:
        - csv_path (str): A path to the csv file
        - transform (Callable, optional): A function to transform the data
        """

        df = pd.read_csv(csv_path)
        self.data = df["text"].tolist()

        if transform is not None:
            self.data = list(filter(lambda x: x != "", map(transform, self.data)))

        self.vocab = Vocabulary(self.data)

        train, test = self.split()
        self.train = np.array([self.vocab.char2idx[char] for char in " ".join(train)])
        self.test = np.array([self.vocab.char2idx[char] for char in " ".join(test)])

    def split(self, train_fraction: float = 0.8) -> tuple[list[str], list[str]]:
        """
        Splits the dataset into train and test data

        Parameters:
        - train_fraction (float, optional): Fraction of data to be used for training. Defaults to 0.8

        Returns:
        - A tuple containing lists of train and test data
        """

        train_data_len = int(len(self.data) * train_fraction)
        return self.data[:train_data_len], self.data[train_data_len:]
