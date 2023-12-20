import unittest

from src.generator.utils.preprocess import clean_string


class TestCleanSentence(unittest.TestCase):
    def test_sentence_trim(self):
        # Arrange
        sentence = "  Hello World  "  # Two spaces around

        # Act
        output = clean_string(sentence)

        # Assert
        self.assertEqual(output, "Hello World")  # Spaces should have been trimmed

    def test_unicode_normalization(self):
        # Arrange
        sentence = "Héllo Wörld"

        # Act
        output = clean_string(sentence)

        # Assert
        self.assertEqual(output, "Hello World")  # Diacritical marks should have been removed

    def test_combined(self):
        # Arrange
        sentence = "  Héllo Wörld  "  # Both: spaces around and diacritical marks

        # Act
        output = clean_string(sentence)

        # Assert
        self.assertEqual(output, "Hello World")  # Spaces should have been trimmed & diacritical marks removed


if __name__ == "__main__":
    unittest.main()
