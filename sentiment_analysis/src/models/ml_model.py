"""
Classical ML Model Module
--------------------------
Model A: TF-IDF + Logistic Regression with GridSearchCV hyperparameter tuning.
Uses scikit-learn pipeline for clean, reproducible training and inference.
"""

import os
import time
import pickle
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SentimentMLModel:
    """
    Classical ML sentiment classifier.
    Wraps TF-IDF + LogisticRegression in a sklearn Pipeline with GridSearchCV tuning.
    """

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(self, config_path: str = "src/config/ml_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.pipeline = None
        self.best_params = None
        self.training_time = None
        self.model_size_bytes = None

    def _build_pipeline(self) -> Pipeline:
        """Construct sklearn Pipeline with TF-IDF and LogisticRegression."""
        cfg_vec = self.config["vectorizer"]
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=cfg_vec["max_features"],
                ngram_range=tuple(cfg_vec["ngram_range"]),
                sublinear_tf=cfg_vec["sublinear_tf"],
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"\w{2,}",
            )),
            ("clf", LogisticRegression(
                random_state=self.config["training"]["random_state"],
                max_iter=500,
            )),
        ])

    def _param_grid(self) -> dict:
        hp = self.config["hyperparameters"]
        return {
            "clf__C": hp["C"],
            "clf__solver": hp["solver"],
        }

    def train(self, X_train: pd.Series, y_train: pd.Series) -> None:
        """Train with GridSearchCV for hyperparameter tuning."""
        logger.info("Building pipeline...")
        base_pipeline = self._build_pipeline()

        cv = StratifiedKFold(
            n_splits=self.config["training"]["cv_folds"],
            shuffle=True,
            random_state=self.config["training"]["random_state"],
        )

        logger.info("Starting GridSearchCV...")
        grid = GridSearchCV(
            base_pipeline,
            param_grid=self._param_grid(),
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1,
        )

        start = time.time()
        grid.fit(X_train, y_train)
        self.training_time = time.time() - start

        self.pipeline = grid.best_estimator_
        self.best_params = grid.best_params_

        logger.info(f"Training complete in {self.training_time:.2f}s")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best CV F1 (macro): {grid.best_score_:.4f}")

    def predict(self, texts) -> np.ndarray:
        """Predict sentiment labels."""
        assert self.pipeline is not None, "Model not trained. Call train() first."
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict(texts)

    def predict_proba(self, texts) -> np.ndarray:
        """Return class probabilities."""
        assert self.pipeline is not None, "Model not trained."
        if isinstance(texts, str):
            texts = [texts]
        return self.pipeline.predict_proba(texts)

    def predict_single(self, text: str) -> dict:
        """Full inference result for one text (used by API)."""
        start = time.time()
        pred = self.predict([text])[0]
        proba = self.predict_proba([text])[0]
        elapsed_ms = (time.time() - start) * 1000
        return {
            "label": int(pred),
            "label_name": self.LABEL_MAP[int(pred)],
            "probabilities": {
                "negative": float(proba[0]),
                "neutral": float(proba[1]),
                "positive": float(proba[2]),
            },
            "inference_time_ms": round(elapsed_ms, 3),
            "model": "classical_ml",
        }

    def evaluate(self, X_test: pd.Series, y_test: pd.Series) -> dict:
        """Full evaluation: accuracy, classification report, confusion matrix."""
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds,
                                        target_names=["negative", "neutral", "positive"],
                                        output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # Measure inference speed
        speeds = []
        for text in X_test[:50]:
            t0 = time.time()
            self.predict([text])
            speeds.append((time.time() - t0) * 1000)

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "avg_inference_ms": float(np.mean(speeds)),
            "training_time_s": self.training_time,
        }

    def save(self, path: str = "models/saved/ml_model.pkl") -> None:
        """Serialize trained pipeline."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        self.model_size_bytes = os.path.getsize(path)
        logger.info(f"Model saved to {path} ({self.model_size_bytes / 1024:.1f} KB)")

    @classmethod
    def load(cls, path: str = "models/saved/ml_model.pkl",
             config_path: str = "src/config/ml_config.yaml") -> "SentimentMLModel":
        """Load a saved pipeline."""
        instance = cls(config_path=config_path)
        with open(path, "rb") as f:
            instance.pipeline = pickle.load(f)
        instance.model_size_bytes = os.path.getsize(path)
        logger.info(f"Model loaded from {path}")
        return instance


def run_training_pipeline(data_path: str = "data/processed/combined_dataset.csv",
                           config_path: str = "src/config/ml_config.yaml",
                           model_save_path: str = "models/saved/ml_model.pkl") -> dict:
    """End-to-end training pipeline for ML model."""
    import sys
    sys.path.insert(0, ".")
    from src.preprocessing.text_preprocessor import TextPreprocessor

    logger.info("=== ML Training Pipeline ===")

    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples")

    # Preprocess
    preprocessor = TextPreprocessor(mode="ml")
    df = preprocessor.process_dataframe(df, text_col="text")
    X = df["cleaned_text_ml"]
    y = df["label"]

    # Split
    with open(config_path) as f:
        config = yaml.safe_load(f)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=config["training"]["random_state"],
        stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train
    model = SentimentMLModel(config_path=config_path)
    model.train(X_train, y_train)

    # Evaluate
    results = model.evaluate(X_test, y_test)
    logger.info(f"\nAccuracy: {results['accuracy']:.4f}")
    logger.info(f"Avg Inference: {results['avg_inference_ms']:.2f}ms")

    # Save
    model.save(model_save_path)

    return {
        "model": model,
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    run_training_pipeline()
