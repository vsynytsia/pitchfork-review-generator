from typing import Iterable

import numpy as np
import torch


def one_hot_encode(arr: np.ndarray, n_labels: int) -> torch.Tensor:
    """
    One-hot encodes each element of a numpy array.

    Parameters:
    - arr (np.ndarray): Input numpy array
    - n_labels (int): Length of the element in the encoded output array

    Returns:
    - Encoded Tensor where each element of the input array is one-hot encoded
    """

    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    return torch.from_numpy(
        one_hot.reshape((*arr.shape, n_labels))
    )


def get_batches(encoded_data: Iterable[int], batch_size: int, seq_len: int):
    """
    Yields batches of specified size from the encoded data.

    Parameters:
    - encoded_data (Iterable[int] or np.ndarray): Encoded data to extract batches from.
    - batch_size (int): Number of sequences per batch.
    - seq_len (int): Length of each sequence.

    Yields:
    - Tuple containing inputs and targets, each of size: `batch_size` * `seq_len`
    """

    total_batch_size = batch_size * seq_len
    n_batches = len(encoded_data) // total_batch_size
    encoded_data = encoded_data[:n_batches * total_batch_size]

    encoded_data = encoded_data.reshape((batch_size, -1))

    for i in range(0, encoded_data.shape[1], seq_len):
        inputs = encoded_data[:, i:i + seq_len]
        targets = np.zeros_like(inputs)

        try:
            targets[:, :-1] = inputs[:, 1:]
            targets[:, -1] = inputs[:, i + seq_len]
        except IndexError:
            targets[:, :-1] = inputs[:, 1:]
            targets[:, -1] = inputs[:, 0]
        yield torch.from_numpy(inputs), torch.from_numpy(targets)
