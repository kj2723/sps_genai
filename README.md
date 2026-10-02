# sps_genai

FastAPI app from the Module 3 class activity (bigram text generation), extended with spaCy word embeddings for Assignment 1.

## Run with Docker
```
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
Open http://localhost:8000/docs

## Endpoints
- POST /generate: `{"start_word": "the", "length": 6}`
- POST /embedding: `{"word": "apple"}` returns the 300-d spaCy embedding
- POST /similarity: `{"word1": "apple", "word2": "car"}`