import logging
from pathlib import Path

import pandas as pd

from models.parsed_review_page import ParsedReviewPage
from src.config import DataFoldersConfig

logger = logging.getLogger(__name__)


def parsed_page_to_json_reviews(parsed_review_pages: list[ParsedReviewPage]) -> list[dict]:
    """
    Convert parsed review pages to JSON format.

    Parameters:
    - parsed_review_pages (list): List of parsed review pages

    Returns:
    - List of reviews in JSON format
    """

    json_reviews = []
    for page in parsed_review_pages:
        for review in page.reviews:
            if review:
                json_reviews.append(
                    review.model_dump()
                )
    return json_reviews


def generate_file_path(start: int, stop: int, folder_path: str) -> Path:
    """
    Generate a file path for saving scraped reviews.

    Parameters:
    - start (int): A number specifying the starting page
    - stop (int): A number specifying the end page
    - folder_path (str): Folder path where the data will be stored

    Returns:
    - A Path object specifying file path for saving data
    """

    raw_data_path = Path(folder_path) / DataFoldersConfig.RAW
    if not raw_data_path.exists():
        raw_data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating {raw_data_path.as_posix()} folder")

    file_name = Path(f"reviews{start}-{stop}.csv")
    return raw_data_path / file_name


def save_to_csv(df: pd.DataFrame, file_path: Path):
    """
    Save a pandas DataFrame to a csv file.

    Parameters:
    - df (DataFrame): A pandas DataFrame
    - file_path (str): A file path where the csv file will be saved
    """

    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Successfully wrote {len(df)} lines to file")
    except Exception as e:
        logger.exception(f"Failed to write DataFrame to csv: {str(e)}")
