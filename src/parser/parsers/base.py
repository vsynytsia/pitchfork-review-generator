import logging
from abc import abstractmethod, ABC

import aiohttp
from bs4 import BeautifulSoup
from fake_headers import Headers

logger = logging.getLogger(__name__)


class BaseParser(ABC):
    """
    An abstract base class that provides a blueprint for all parsers.
    Methods defined in this class should be overridden by any derived class.
    Contains staticmethod to retrieve BeautifulSoup object for the URL.
    """

    @staticmethod
    async def get_soup(url: str) -> BeautifulSoup:
        """
       Fetch page data from a URL, convert it to BeautifulSoup object.

       Parameters:
       - url (str): URL of the webpage

       Returns:
       - BeautifulSoup object
       """
        headers = Headers(os="win", browser="chrome").generate()
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                status_code = response.status

                if status_code != 200:
                    logger.exception(f"Status code {status_code} occurred when making a request to {url}")

                page = await response.text()
                return BeautifulSoup(page, "html.parser")

    @abstractmethod
    async def parse(self):
        raise NotImplementedError
