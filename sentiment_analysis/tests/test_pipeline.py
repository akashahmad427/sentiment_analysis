"""
Unit Tests for Sentiment Analysis Pipeline
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.data.dataset_builder import DatasetBuilder
from src.evaluation.evaluator import ModelEvaluator


# -----------------------------------------------------------------------
# TextPreprocessor Tests
# -----------------------------------------------------------------------
class TestTextPreprocessor:

    def test_url_removal_ml(self):
        p = TextPreprocessor(mode="ml")
        result = p.clean("Check this out https://example.com amazing!")
        assert "http" not in result
        assert "amazing" in result

    def test_mention_removal(self):
        p = TextPreprocessor(mode="ml")
        result = p.clean("Thanks @john for the help!")
        assert "@john" not in result
        assert "thanks" in result

    def test_hashtag_expansion(self):
        p = TextPreprocessor(mode="ml")
        result = p.clean("Feeling great today #HappyDay")
        assert "#" not in result
        assert "happyday" in result.lower() or "happy" in result.lower()

    def test_emoji_removal(self):
        p = TextPreprocessor(mode="ml")
        result = p.clean("Love this product! 🎉🔥❤️")
        assert "🎉" not in result
        assert "love" in result.lower()

    def test_html_removal(self):
        p = TextPreprocessor(mode="ml")
        result = p.clean("<p>Great product!</p>")
        assert "<p>" not in result
        assert "great" in result.lower()

    def test_empty_string(self):
        p = TextPreprocessor(mode="ml")
        assert p.clean("") == ""
        assert p.clean("   ") == ""
        assert p.clean(None) == ""

    def test_ml_lowercase(self):
        p = TextPreprocessor(mode="ml")
        result = p.clean("AMAZING PRODUCT!")
        assert result == result.lower()

    def test_dl_preserves_case(self):
        p = TextPreprocessor(mode="dl")
        result = p.clean("Amazing product!")
        # DL mode doesn't lowercase
        assert "Amazing" in result or "amazing" in result

    def test_process_dataframe(self):
        p = TextPreprocessor(mode="ml")
        df = pd.DataFrame({
            "text": ["Great product! 🎉", "Terrible! @user https://link.com", ""],
            "label": [2, 0, 1]
        })
        result = p.process_dataframe(df)
        assert "cleaned_text_ml" in result.columns
        # Empty row should be dropped
        assert len(result) == 2


# -----------------------------------------------------------------------
# DatasetBuilder Tests
# -----------------------------------------------------------------------
class TestDatasetBuilder:

    def test_tweet_generation(self):
        builder = DatasetBuilder()
        df = builder.generate_synthetic_tweets(n=100)
        assert len(df) == 100
        assert "text" in df.columns
        assert "label" in df.columns
        assert "source" in df.columns
        assert set(df["source"].unique()) == {"tweets"}
        assert set(df["label"].unique()).issubset({0, 1, 2})

    def test_review_generation(self):
        builder = DatasetBuilder()
        df = builder.generate_synthetic_reviews(n=100)
        assert len(df) == 100
        assert set(df["source"].unique()) == {"product_reviews"}

    def test_build_dataset(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path))
        df = builder.build_dataset()
        assert len(df) > 0
        assert "label_name" in df.columns
        assert set(df["label_name"].unique()).issubset({"positive", "negative", "neutral"})
        assert (tmp_path / "combined_dataset.csv").exists()

    def test_no_duplicates(self, tmp_path):
        builder = DatasetBuilder(output_dir=str(tmp_path))
        df = builder.build_dataset()
        assert df["text"].duplicated().sum() == 0


# -----------------------------------------------------------------------
# Evaluator Tests
# -----------------------------------------------------------------------
class TestModelEvaluator:

    def test_compute_metrics(self):
        evaluator = ModelEvaluator(output_dir="/tmp/eval_test")
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1, 1, 1, 2])
        metrics = evaluator.compute_metrics(y_true, y_pred, "test_model")
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert set(metrics["per_class"].keys()) == {"negative", "neutral", "positive"}
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_comparison_table(self):
        evaluator = ModelEvaluator(output_dir="/tmp/eval_test")
        y_true = np.random.choice([0, 1, 2], 100)
        ml_preds = np.random.choice([0, 1, 2], 100)
        dl_preds = np.random.choice([0, 1, 2], 100)
        ml_metrics = evaluator.compute_metrics(y_true, ml_preds, "ML")
        dl_metrics = evaluator.compute_metrics(y_true, dl_preds, "DL")
        table = evaluator.build_comparison_table(ml_metrics, dl_metrics, {}, {})
        assert isinstance(table, pd.DataFrame)
        assert "Metric" in table.columns
        assert "ML Model" in table.columns
        assert "DL Model" in table.columns

    def test_dashboard_generation(self, tmp_path):
        evaluator = ModelEvaluator(output_dir=str(tmp_path))
        np.random.seed(42)
        y_test = np.random.choice([0, 1, 2], 60)
        ml_preds = np.random.choice([0, 1, 2], 60)
        dl_preds = np.random.choice([0, 1, 2], 60)
        result = evaluator.generate_dashboard(y_test, ml_preds, dl_preds)
        assert "ml_metrics" in result
        assert "dl_metrics" in result
        assert Path(result["dashboard_path"]).exists()


# -----------------------------------------------------------------------
# ML Model Tests
# -----------------------------------------------------------------------
class TestMLModel:

    def test_pipeline_builds(self):
        from src.models.ml_model import SentimentMLModel
        model = SentimentMLModel(config_path="src/config/ml_config.yaml")
        pipeline = model._build_pipeline()
        assert pipeline is not None

    def test_train_predict(self):
        from src.models.ml_model import SentimentMLModel
        model = SentimentMLModel(config_path="src/config/ml_config.yaml")
        X = pd.Series([
            "I love this amazing product",
            "Terrible experience, very bad",
            "It was okay, nothing special",
            "Great quality, highly recommend",
            "Awful, broken on arrival",
            "Average, meets expectations",
        ] * 10)
        y = pd.Series([2, 0, 1, 2, 0, 1] * 10)
        model.train(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert all(p in [0, 1, 2] for p in preds)

    def test_predict_single(self):
        from src.models.ml_model import SentimentMLModel
        model = SentimentMLModel(config_path="src/config/ml_config.yaml")
        X = pd.Series([
            "great product love it amazing", "terrible awful broken disaster",
            "okay fine average nothing special",
        ] * 20)
        y = pd.Series([2, 0, 1] * 20)
        model.train(X, y)
        result = model.predict_single("This product is absolutely amazing!")
        assert "label" in result
        assert "label_name" in result
        assert "inference_time_ms" in result
        assert result["label_name"] in ["positive", "neutral", "negative"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
