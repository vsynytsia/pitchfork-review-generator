import asyncio
import logging

import pandas as pd

from src.parser.models import ParsedReviewPage
from src.parser.parsers import ReviewPageParser
from utils import parsed_page_to_json_reviews, generate_file_path, save_to_csv

logger = logging.getLogger(__name__)


class PitchforkCrawler:
    """Asynchronous pitchfork.com scraper"""

    def __init__(self, n_pages: int, start_page: int = 1, sleep_time: float = 0.2):
        """
       Initialize a new PitchforkCrawler instance.

       Parameters:
       - n_pages (int): The number of pages to be crawled
       - start_page (int, optional): The start page number. Defaults to 1
       - sleep_time (float, optional): The sleep time between requests in seconds. Defaults to 0.2
       """

        self.start = start_page
        self.stop = start_page + n_pages
        self.sleep_time = sleep_time

    async def crawl(self):
        """Crawl the pages and save result to CSV file"""

        parsed_pages = await self._parse()
        json_reviews = parsed_page_to_json_reviews(parsed_pages)

        df = pd.DataFrame(json_reviews)
        file_path = generate_file_path(self.start, self.stop, folder_path="../../")
        save_to_csv(df, file_path)

    async def _parse(self) -> list[ParsedReviewPage]:
        tasks = []
        for i in range(self.start, self.stop):
            review = ReviewPageParser(page_number=i)
            tasks.append(
                review.parse()
            )
            await asyncio.sleep(self.sleep_time)

        return await asyncio.gather(*tasks)
