import logging
from typing import Any, Callable, Optional

import bs4

from src.parser.models import ParsedReview
from .base import BaseParser


def handle_absence(default_value: Any) -> Callable:
    """
    Decorator factory for handling absence of value in function's result.
    When decorated function raises an IndexError, returns default value.

    Parameters:
    - default_value (Any): A value to return in case of IndexError

    Returns:
    - Callable decorator
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except IndexError:
                return default_value

        return wrapper

    return decorator


handle_text_absence = handle_absence(default_value="")
handle_float_absence = handle_absence(default_value=0.0)


class ReviewParser(BaseParser):
    """Parser for individual review from pitchfork."""

    def __init__(self, url: str):
        """
        Initialize a new instance of ReviewParser.

        Parameters:
        - url (str): URL of the review.
        """

        super().__init__()
        self.url = url
        self.soup = None

    async def parse(self) -> Optional[ParsedReview]:
        self.soup = await self.get_soup(self.url)

        try:
            summary = self.parse_summary()
            review_text = self.parse_review_text()
            score = self.parse_score()

            misc_info = self._parse_misc_info()
            label, genre = misc_info.get("Label", ""), misc_info.get("Genre", "")

            artist_name = self.parse_artist_name()
            album_name = self.parse_album_name()
        except Exception as ex:
            logging.exception(f"Error parsing {self.url}: {ex}")
            return

        return ParsedReview(
            url=self.url,
            artist_name=artist_name,
            label=label,
            album_name=album_name,
            genre=genre,
            summary=summary,
            review_text=review_text,
            score=score,
        )

    def parse_artist_name(self) -> str:
        artist_name_tag = self._select_one_containing("div[class*='SplitScreenContentHeaderArtist']")
        return self._get_text(artist_name_tag)

    def parse_album_name(self) -> str:
        album_name_tag = self._select_one_containing("h1[class*='SplitScreenContentHeaderHed']")
        return self._get_text(album_name_tag)

    def parse_summary(self) -> str:
        summary_tag = self._select_one_containing("div[class*='SplitScreenContentHeaderDekDown']")
        return self._get_text(summary_tag)

    def parse_review_text(self) -> str:
        return " ".join(
            (paragraph.text for paragraph in self._select_one_containing("div[class*='body__inner-container']"))
        )

    def parse_score(self) -> float:
        score_tag = self._select_one_containing("div[class*='ScoreCircle']")
        return self._get_float(score_tag)

    def _select_one_containing(self, selector: str) -> list[bs4.element.Tag]:
        return self.soup.select(selector=selector)

    def _parse_misc_info(self) -> dict[str, str]:
        misc_info_tag = self._select_one_containing("li[class*='InfoSliceListItem']")
        misc_info = {}

        for tag in misc_info_tag:
            split_tag = tag.text.split(":")
            misc_info[split_tag[0]] = split_tag[1]

        return misc_info

    @handle_text_absence
    def _get_text(self, tag: bs4.element.Tag) -> str:
        return tag[0].text

    @handle_float_absence
    def _get_float(self, tag: bs4.element) -> float:
        return float(tag[0].text)
