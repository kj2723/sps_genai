from fastapi import FastAPI
from pydantic import BaseModel
from app.embedding_model import calculate_embedding, calculate_similarity

app = FastAPI()


class EmbeddingRequest(BaseModel):
    word: str


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV!"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding.tolist()}


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": float(similarity)}