import asyncio
from urllib.parse import urljoin, urlencode

from src.config import UrlConfig
from src.parser.models import ParsedReviewPage
from src.parser.parsers import ReviewParser
from .base import BaseParser


class ReviewPageParser(BaseParser):
    """
    A parser for the review page. It takes a page number, generates the appropriate
    URL, fetches and parses the page content.
    """

    def __init__(self, page_number: int):
        """
        Initialize ReviewPageParser with a specific page number.

        Parameters:
        - page_number (int): The page number to be crawled and parsed
        """

        super().__init__()
        self.page_number = page_number
        self.base_page_url = self._construct_base_page_url(page_number)
        self.soup = None

    async def parse(self) -> ParsedReviewPage:
        """
        Parse the given review page and return the results in the form of a ParsedReviewPage object.

        Returns:
        - Parsed details from the review page
        """

        self.soup = await self.get_soup(self.base_page_url)
        parsed_reviews = await self._parse_reviews()
        return ParsedReviewPage(
            page_number=self.page_number,
            url=self.base_page_url,
            reviews=parsed_reviews
        )

    def fetch_page_urls(self) -> list[str]:
        """
        Find and all review URLs on the page.

        Returns:
        - A list of review URLs
        """

        page_tree = self.soup.find_all("div", class_="review")
        hrefs = [page_item.find("a", class_="review__link").get("href") for page_item in page_tree]
        return [urljoin(UrlConfig.PITCHFORK, href) for href in hrefs]

    async def _parse_reviews(self) -> list:
        page_urls = self.fetch_page_urls()
        tasks = (ReviewParser(page_url).parse() for page_url in page_urls)
        return await asyncio.gather(*tasks)

    @staticmethod
    def _construct_base_page_url(page_number) -> str:
        return f"{UrlConfig.ALBUM_REVIEWS}/?{urlencode({'page': page_number})}"
