# sps_genai

A FastAPI application that serves text generation, word embedding, and image classification models.

## Getting Started

### Run with Docker

```
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```

Then open http://localhost:8000/docs to view and test all endpoints.

### Run Locally

```
uv sync
uv run uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs

## Endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/generate` | POST | `{"start_word": "the", "length": 6}` | Text generated with a bigram model |
| `/embedding` | POST | `{"word": "apple"}` | 300-dimensional spaCy word embedding |
| `/similarity` | POST | `{"word1": "apple", "word2": "car"}` | Similarity score between two words |
| `/classify_image` | POST | Image file upload | Predicted CIFAR-10 class and confidence |

## Assignment 1: Word Embeddings

Word embeddings are generated with the spaCy `en_core_web_lg` model, which provides 300-dimensional word vectors. The `/embedding` endpoint returns the vector for a given word, and `/similarity` returns the cosine similarity between two words.

Example:

```
curl -X POST http://localhost:8000/similarity \
  -H "Content-Type: application/json" \
  -d '{"word1": "apple", "word2": "car"}'
```

## Assignment 2: CNN Image Classifier

### Architecture

| Layer | Output Shape |
|---|---|
| Input (RGB image) | 64 × 64 × 3 |
| Conv2D (16 filters, 3×3, stride 1, padding 1) + ReLU | 64 × 64 × 16 |
| MaxPool (2×2, stride 2) | 32 × 32 × 16 |
| Conv2D (32 filters, 3×3, stride 1, padding 1) + ReLU | 32 × 32 × 32 |
| MaxPool (2×2, stride 2) | 16 × 16 × 32 |
| Flatten | 8192 |
| Fully Connected + ReLU | 100 |
| Fully Connected | 10 |

### Training

- **Dataset:** CIFAR-10, with images resized from 32×32 to 64×64 to match the input specification
- **Setup:** 5 epochs, Adam optimizer (learning rate 0.001), cross-entropy loss, batch size 64
- **Test accuracy:** 66.4%

To retrain the model:

```
uv run python train_cnn_model.py
```

### Example

```
curl -X POST -F "file=@cat.png" http://localhost:8000/classify_image
```

Response:

```json
{"class": "cat", "confidence": 0.6111}
```

Supported classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Project Structure

```
sps_genai/
├── main.py                 FastAPI application and endpoints
├── app/
│   ├── bigram_model.py     Bigram text generation
│   ├── embedding_model.py  spaCy embeddings and similarity
│   ├── cnn_model.py        CNN architecture and inference
│   └── cnn_weights.pth     Trained CNN weights
├── train_cnn_model.py      CNN training script
├── Dockerfile
└── pyproject.toml
```