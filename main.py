from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding, calculate_similarity
from fastapi import File, UploadFile, HTTPException
from app.cnn_model import CNNClassifier

import io
import torch
from fastapi.responses import Response
from torchvision.utils import save_image
from helper_lib.model import get_model

app = FastAPI()
cnn_classifier = CNNClassifier()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str

class SimilarityRequest(BaseModel):
    word1: str
    word2: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding.tolist()}

@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    similarity = calculate_similarity(request.word1, request.word2)
    return {"word1": request.word1, "word2": request.word2, "similarity": float(similarity)}

@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    return cnn_classifier.predict(await file.read())


gan = get_model("GAN")
gan.generator.load_state_dict(torch.load("app/gan_generator.pth", map_location="cpu"))
gan.generator.eval()


@app.get("/generate_digits")
def generate_digits(n: int = 16):
    """Generate n handwritten digits with the trained GAN, returned as a PNG grid."""
    with torch.no_grad():
        imgs = gan.generator(torch.randn(n, gan.z_dim))
    buf = io.BytesIO()
    save_image(imgs, buf, format="png", nrow=4, normalize=True)
    return Response(content=buf.getvalue(), media_type="image/png")