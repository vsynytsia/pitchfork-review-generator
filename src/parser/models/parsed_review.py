from pydantic import BaseModel, Field


class ParsedReview(BaseModel):
    url: str = Field(
        description="The URL of the album"
    )
    artist_name: str = Field(
        description="The name of the artist"
    )
    label: str = Field(
        description="The name of the label"
    )
    album_name: str = Field(
        description="The name of the album"
    )
    genre: str = Field(
        description="Genre of the album"
    )
    summary: str = Field(
        description="Summary of the album review"
    )
    review_text: str = Field(
        description="Text of the album review"
    )
    score: float = Field(
        description="Score of the album"
    )
