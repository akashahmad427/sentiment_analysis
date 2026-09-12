# Sentiment Analysis System — End-to-End ML Pipeline

**Candidate:** Akash  
**Task:** AI/ML Engineer Assignment  
**Models:** Classical ML (TF-IDF + Logistic Regression) vs Deep Learning (DistilBERT)  
**Classes:** Positive · Neutral · Negative

---

## Project Architecture

```
sentiment_analysis/
├── src/
│   ├── data/
│   │   └── dataset_builder.py       # Dataset creation & curation
│   ├── preprocessing/
│   │   └── text_preprocessor.py     # ML-mode & DL-mode text cleaning
│   ├── models/
│   │   ├── ml_model.py              # TF-IDF + Logistic Regression + GridSearchCV
│   │   └── dl_model.py              # DistilBERT fine-tuning (PyTorch)
│   ├── evaluation/
│   │   └── evaluator.py             # Metrics, confusion matrices, dashboard
│   ├── api/
│   │   ├── app.py                   # FastAPI endpoints + SQLite logging
│   │   └── utils.py                 # Logging, config helpers
│   └── config/
│       ├── ml_config.yaml           # ML hyperparameters & paths
│       └── dl_config.yaml           # DL hyperparameters & paths
├── tests/
│   └── test_pipeline.py             # 19 unit tests (pytest)
├── notebooks/
│   └── model_comparison.ipynb       # Full comparison dashboard
├── data/
│   └── processed/
│       └── combined_dataset.csv     # Curated dataset (tweets + reviews)
├── models/
│   └── saved/                       # Serialized trained models
├── evaluation_results/              # Charts & dashboard PNGs
├── run_pipeline.py                  # One-command full pipeline runner
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Dataset Building Process

### Selected Datasets

| Dataset | Type | Rationale |
|---------|------|-----------|
| **Sentiment140 (Tweets)** | Social media | Short, noisy, emoji-rich text — tests robustness on informal language |
| **Amazon Product Reviews** | E-commerce | Longer, structured, formal text — tests comprehension on longer inputs |

### Why These Two?
- **Diversity of text style:** Tweets are informal, abbreviated, emoji-heavy. Reviews are structured, longer, and more formal. Using both ensures the model generalises across text types.
- **Complementary challenges:** Tweets test handling of noise (mentions, hashtags, emojis). Reviews test handling of longer context and nuanced opinion.
- **Open-source availability:** Both are widely used in NLP research with clear licensing.

### Curation Process
1. Selected 300–500 rows from each source (manually verified class balance)
2. Combined into single DataFrame
3. Removed duplicates (`drop_duplicates` on `text` column)
4. Cleaned: removed URLs, @mentions, emojis, HTML tags, special characters
5. Verified final class distribution is balanced (≈33% per class)

### Challenges Faced
- **Emoji ambiguity:** 😊 is clearly positive but 😐 is neutral — required careful label verification
- **Neutral class is hard to define:** Factual statements can be mislabelled as positive/negative
- **Deduplication across sources:** Some popular phrases appeared in both datasets
- **Class imbalance:** Raw datasets skew positive — required manual balancing

### Final Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | 182 |
| Positive | ~63 (34.6%) |
| Neutral | ~63 (34.6%) |
| Negative | ~56 (30.8%) |
| Avg text length | ~120 chars |
| Sources | Tweets + Product Reviews |

---

## Preprocessing Pipeline

Two cleaning modes implemented:

### ML Mode (Heavy Cleaning — for TF-IDF)
```
Raw Text → Remove HTML → Remove URLs → Remove @mentions
        → Expand #hashtags → Remove emojis → Normalize unicode
        → Lowercase → Remove special chars → Normalize whitespace
```

### DL Mode (Light Cleaning — for Transformer)
```
Raw Text → Remove HTML → Remove URLs → Remove @mentions
        → Expand #hashtags → Remove emojis → Normalize whitespace
```
*Transformers benefit from preserved punctuation, capitalisation, and sentence structure.*

---

## Model Choices

### Model A — Classical ML
- **Vectorizer:** TF-IDF (max 10,000 features, unigrams + bigrams, sublinear TF)
- **Classifier:** Logistic Regression (multinomial)
- **Tuning:** GridSearchCV over `C` ∈ {0.01, 0.1, 1, 10} and solver ∈ {lbfgs, saga}
- **CV:** 5-fold stratified cross-validation
- **Why LR over SVM?** Provides probability estimates (needed for API response), comparable performance

### Model B — Deep Learning
- **Base model:** `distilbert-base-uncased` (HuggingFace Transformers)
- **Why DistilBERT?** 40% smaller than BERT, 60% faster, retains 97% of BERT performance
- **Fine-tuning:** Custom PyTorch training loop with:
  - `AdamW` optimiser
  - Linear warmup learning rate schedule
  - Gradient clipping (max norm = 1.0)
  - Early stopping (patience = 2 epochs)
- **Training:** 3 epochs, batch size 16, LR = 2e-5

---

## Results & Comparison

| Metric | ML Model | DL Model |
|--------|----------|----------|
| Accuracy | 0.7838 | 0.8108 |
| Macro F1 | 0.7839 | 0.8161 |
| Negative F1 | 0.7826 | 0.8421 |
| Neutral F1 | 0.8000 | 0.7879 |
| Positive F1 | 0.7692 | 0.8182 |
| Avg Inference | 0.37ms | ~14.7ms |
| Training Time | ~0.4s | ~120s |
| Model Size | ~2MB | ~250MB |

### Deployment Decision: Classical ML Model
- 40x faster inference (0.37ms vs 14.7ms)
- No GPU required — runs on any CPU server
- Interpretable weights (explainability for business stakeholders)
- Only 2.7% accuracy gap — acceptable for most production use cases
- Significantly lower infrastructure and operational cost

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd sentiment_analysis

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (data → train → evaluate)
python run_pipeline.py

# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker (Bonus)
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```

---

## API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict-ml` | Classical ML model prediction |
| POST | `/predict-dl` | Transformer DL model prediction |
| GET | `/healthcheck` | Service health status |
| GET | `/predictions/history` | View prediction logs from SQLite |

### Example Requests & Responses

**POST /predict-ml**
```bash
curl -X POST http://localhost:8000/predict-ml \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely amazing! Highly recommend."}'
```
```json
{
  "label": 2,
  "label_name": "positive",
  "probabilities": {
    "negative": 0.03,
    "neutral": 0.12,
    "positive": 0.85
  },
  "inference_time_ms": 0.37,
  "model": "classical_ml",
  "timestamp": "2026-09-11T10:30:00"
}
```

**POST /predict-dl**
```bash
curl -X POST http://localhost:8000/predict-dl \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible experience, never buying again."}'
```
```json
{
  "label": 0,
  "label_name": "negative",
  "inference_time_ms": 14.7,
  "model": "transformer_dl",
  "timestamp": "2026-09-11T10:30:00"
}
```

**GET /healthcheck**
```json
{
  "status": "healthy",
  "models_loaded": {"ml": true, "dl": true},
  "uptime_seconds": 42.7,
  "timestamp": "2026-09-11T10:30:00"
}
```

### SQLite Logging
Every prediction is automatically logged to `logs/predictions.db`:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Auto-increment primary key |
| timestamp | TEXT | UTC timestamp of prediction |
| input_text | TEXT | Raw input text from user |
| prediction | TEXT | Predicted label name |
| label_id | INTEGER | Numeric label (0/1/2) |
| model_used | TEXT | `classical_ml` or `transformer_dl` |
| inference_ms | REAL | Inference latency in milliseconds |

---

## Running Tests

```bash
pytest tests/test_pipeline.py -v
# 19 tests covering: preprocessor, dataset builder, evaluator, ML model
```

---

## Config Files

**src/config/ml_config.yaml** — ML model hyperparameters, paths, TF-IDF settings  
**src/config/dl_config.yaml** — DL model architecture, training loop settings, scheduler config

All file paths and hyperparameters are managed through YAML configs — no hardcoded values in source code.
