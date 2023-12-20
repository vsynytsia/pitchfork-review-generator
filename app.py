import pickle
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from src.generator.utils.dataset import Vocabulary

from src.generator.models.text import TextGenerationModel

app = FastAPI()


class ReviewInput(BaseModel):
    text_length: int = Field(..., gt=0)
    prime: Optional[str] = None


class ReviewOutput(BaseModel):
    generated_review: str


@app.post("/generate_review", response_model=ReviewOutput)
async def generate_review(request: Request) -> ReviewOutput:
    json_request = await request.json()

    with open("models/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    text_model = TextGenerationModel.from_checkpoint(
        checkpoint_path="models/lstm.pt",
        vocab=vocab
    )
    generated_review = text_model.generate_text(text_length=json_request["text_length"], prime=json_request["prime"])
    print(generated_review)
    return ReviewOutput(generated_review=generated_review)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
