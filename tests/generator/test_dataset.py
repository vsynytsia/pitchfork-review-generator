import unittest
from unittest.mock import patch

import pandas as pd

from src.generator.utils.dataset import Dataset, Vocabulary


class TestVocabulary(unittest.TestCase):
    def test_vocabulary(self):
        # Arrange
        data = ["hello", "world"]

        # Act
        vocab = Vocabulary(data)

        # Assert
        self.assertEqual(vocab.size, 8)


class TestDataset(unittest.TestCase):
    @patch("pandas.read_csv")
    def test_dataset(self, mock_read_csv):
        # Arrange
        data = pd.DataFrame({"text": ["hello", "world", "python", "programming"]})
        mock_read_csv.return_value = data

        # Act
        dataset = Dataset("dummy_path.csv")
        print(dataset.train, len(dataset.test))

        # Assert
        self.assertEqual(len(dataset.train), 18)
        self.assertEqual(len(dataset.test), 11)

    @patch("pandas.read_csv")
    def test_dataset_with_transform(self, mock_read_csv):
        # Arrange
        data = pd.DataFrame({"text": ["hello", "world", "python", "programming"]})
        mock_read_csv.return_value = data
        transform = lambda x: x.upper()

        # Act
        dataset = Dataset("dummy_path.csv", transform)

        # Assert
        self.assertEqual(len(dataset.train), 18)
        self.assertEqual(len(dataset.test), 11)
        self.assertEqual(dataset.data[0], "HELLO")

    @patch("pandas.read_csv")
    def test_split(self, mock_read_csv):
        # Arrange
        data = pd.DataFrame({"text": ["hello", "world", "python", "programming"]})
        mock_read_csv.return_value = data

        # Act
        dataset = Dataset("dummy_path.csv")
        train, test = dataset.split(0.5)

        # Assert
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)


if __name__ == "__main__":
    unittest.main()
