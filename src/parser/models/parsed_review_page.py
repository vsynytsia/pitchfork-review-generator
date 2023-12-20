from typing import Optional

from pydantic import BaseModel, Field

from .parsed_review import ParsedReview


class ParsedReviewPage(BaseModel):
    page_number: int = Field(
        description="Number of the page"
    )
    url: str = Field(
        description="Page URL"
    )
    reviews: list[Optional[ParsedReview]] = Field(
        description="List of parsed reviews from the page"
    )
