import unittest
from unittest.mock import Mock, patch
import torch
from torch import nn

from src.generator.utils.checkpoint import save_checkpoint, load_checkpoint, get_checkpoint
from src.generator.utils.dataset import Vocabulary


class TestCheckpoint(unittest.TestCase):
    @patch("torch.save")
    def test_save_checkpoint(self, mock_save):
        # Arrange
        model = nn.Linear(2, 2)
        vocab = Vocabulary(["hello", "world"])
        epoch = 1
        checkpoint_folder = "./"

        # Act
        save_checkpoint(checkpoint_folder, model, vocab, epoch)

        # Assert
        mock_save.assert_called()

    @patch("torch.load")
    def test_load_checkpoint(self, mock_load):
        # Arrange
        checkpoint_path = "./checkpoint-1.pt"

        # Act
        load_checkpoint(checkpoint_path)

        # Assert
        mock_load.assert_called()

    def test_get_checkpoint(self):
        # Arrange
        model = nn.Linear(2, 2)
        vocab = Vocabulary(["hello", "world"])
        epoch = 1
        optimizer = Mock()
        optimizer.state_dict.return_value = {"lr": 0.01}

        # Act
        state_dict = get_checkpoint(model, epoch, vocab, optimizer=optimizer)

        # Assert
        self.assertIn("epoch", state_dict)
        self.assertIn("model_state_dict", state_dict)
        self.assertIn("vocab", state_dict)
        self.assertIn("optimizer", state_dict)
        self.assertEqual(state_dict["epoch"], epoch)
        self.assertEqual(state_dict["vocab"], vocab)
        self.assertEqual(state_dict["optimizer"], {"lr": 0.01})


if __name__ == '__main__':
    unittest.main()
