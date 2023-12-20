import unittest

import numpy as np
import torch

from src.generator.utils.batches import get_batches, one_hot_encode


class TestOneHotEncode(unittest.TestCase):
    def test_one_hot_encode(self):
        # Arrange
        arr = np.array([[0, 1], [2, 3]])
        n_labels = 5

        # Act
        result = one_hot_encode(arr, n_labels)

        # Assert
        expected_result = torch.tensor([[[1., 0., 0., 0., 0.],
                                        [0., 1., 0., 0., 0.]],
                                        [[0., 0., 1., 0., 0.],
                                        [0., 0., 0., 1., 0.]]])
        self.assertTrue(torch.equal(result, expected_result))


class TestGetBatches(unittest.TestCase):
    def test_get_batches(self):
        # Arrange
        encoded_data = np.array([0, 1, 2, 3, 4, 5, 6])
        batch_size = 2
        seq_len = 3

        # Act
        batches = list(get_batches(encoded_data, batch_size, seq_len))

        # Assert
        expected_batches = [(torch.tensor([[0, 1, 2], [3, 4, 5]]),
                             torch.tensor([[1, 2, 0], [4, 5, 3]]))]
        for batch, expected_batch in zip(batches, expected_batches):
            self.assertTrue(torch.equal(batch[0], expected_batch[0]))
            self.assertTrue(torch.equal(batch[1], expected_batch[1]))


if __name__ == '__main__':
    unittest.main()
