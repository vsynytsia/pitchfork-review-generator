import re

import unicodedata


def clean_string(string: str) -> str:
    """
    Preprocess the string: unicode normalize and keep only english letters and numbers

    Parameters:
    - sentence (str): String to preprocess

    Returns:
    - Cleaned string
    """

    string = unicodedata.normalize("NFKD", string).encode("ascii", "ignore").decode("utf-8", "ignore")
    return re.sub(r'[^A-Za-z0-9\s.,!?;:\"\']+', "", string).strip()
