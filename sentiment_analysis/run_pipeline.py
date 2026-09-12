"""
End-to-End Pipeline Runner
---------------------------
Runs the complete sentiment analysis pipeline:
  1. Build dataset
  2. Train ML model
  3. Train DL model
  4. Generate evaluation dashboard
  5. Save all artifacts

Usage:
    python run_pipeline.py
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    BASE = Path(__file__).parent
    DATA_DIR = BASE / "data" / "processed"
    MODEL_DIR = BASE / "models" / "saved"
    EVAL_DIR = BASE / "evaluation_results"
    ML_CFG = str(BASE / "src" / "config" / "ml_config.yaml")
    DL_CFG = str(BASE / "src" / "config" / "dl_config.yaml")

    for d in [DATA_DIR, MODEL_DIR, EVAL_DIR, BASE / "logs"]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # STEP 1: Build Dataset                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Building Dataset")
    logger.info("="*60)
    from src.data.dataset_builder import DatasetBuilder
    builder = DatasetBuilder(output_dir=str(DATA_DIR))
    df = builder.build_dataset()
    data_path = str(DATA_DIR / "combined_dataset.csv")
    logger.info(f"Dataset ready: {len(df)} samples at {data_path}")

    # ------------------------------------------------------------------ #
    # STEP 2: Preprocessing                                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Preprocessing")
    logger.info("="*60)
    from src.preprocessing.text_preprocessor import TextPreprocessor

    ml_prep = TextPreprocessor(mode="ml")
    dl_prep = TextPreprocessor(mode="dl")
    df_ml = ml_prep.process_dataframe(df, "text")
    df_dl = dl_prep.process_dataframe(df, "text")

    import yaml
    with open(ML_CFG) as f:
        ml_cfg = yaml.safe_load(f)
    rs = ml_cfg["training"]["random_state"]
    ts = ml_cfg["training"]["test_size"]

    X_ml = df_ml["cleaned_text_ml"]
    X_dl = df_dl["cleaned_text_dl"]
    y = df_ml["label"]

    X_ml_train, X_ml_test, y_train, y_test = train_test_split(X_ml, y, test_size=ts, random_state=rs, stratify=y)
    X_dl_train, X_dl_test, _, _ = train_test_split(X_dl, y, test_size=ts, random_state=rs, stratify=y)

    # ------------------------------------------------------------------ #
    # STEP 3: Train ML Model                                               #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Training Classical ML Model")
    logger.info("="*60)
    from src.models.ml_model import SentimentMLModel
    ml_model = SentimentMLModel(config_path=ML_CFG)
    ml_model.train(X_ml_train, y_train)
    ml_results = ml_model.evaluate(X_ml_test, y_test)
    ml_model.save(str(MODEL_DIR / "ml_model.pkl"))
    ml_preds = ml_model.predict(X_ml_test)
    logger.info(f"ML Accuracy: {ml_results['accuracy']:.4f}")

    # ------------------------------------------------------------------ #
    # STEP 4: Train DL Model                                               #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Training DL Model (DistilBERT)")
    logger.info("="*60)
    from src.models.dl_model import SentimentDLModel
    dl_model = SentimentDLModel(config_path=DL_CFG)

    with open(DL_CFG) as f:
        dl_cfg = yaml.safe_load(f)

    X_dl_tv, X_dl_test_final, y_tv, y_test_dl = train_test_split(
        X_dl_train, y_train, test_size=0.1, random_state=rs, stratify=y_train
    )
    X_dl_t, X_dl_v, y_dl_t, y_dl_v = train_test_split(
        X_dl_tv, y_tv, test_size=0.1, random_state=rs, stratify=y_tv
    )
    dl_model.train(X_dl_t, y_dl_t, X_dl_v, y_dl_v)
    dl_results = dl_model.evaluate(X_dl_test, y_test)
    dl_model.save(str(MODEL_DIR / "dl_model"))
    dl_preds = dl_model.predict(list(X_dl_test))
    logger.info(f"DL Accuracy: {dl_results['accuracy']:.4f}")

    # ------------------------------------------------------------------ #
    # STEP 5: Generate Evaluation Dashboard                                #
    # ------------------------------------------------------------------ #
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Generating Evaluation Dashboard")
    logger.info("="*60)
    from src.evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(output_dir=str(EVAL_DIR))
    eval_output = evaluator.generate_dashboard(
        y_test=list(y_test),
        ml_preds=ml_preds,
        dl_preds=dl_preds,
        ml_extra={
            "avg_inference_ms": ml_results["avg_inference_ms"],
            "training_time_s": ml_results["training_time_s"],
        },
        dl_extra={
            "avg_inference_ms": dl_results["avg_inference_ms"],
            "training_time_s": dl_results["training_time_s"],
        },
        save_path=str(EVAL_DIR / "model_comparison_dashboard.png"),
    )

    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"  Dataset:    {data_path}")
    logger.info(f"  ML Model:   {MODEL_DIR / 'ml_model.pkl'}")
    logger.info(f"  DL Model:   {MODEL_DIR / 'dl_model'}")
    logger.info(f"  Dashboard:  {EVAL_DIR / 'model_comparison_dashboard.png'}")
    logger.info(f"\n  ML Accuracy: {ml_results['accuracy']:.4f} | DL Accuracy: {dl_results['accuracy']:.4f}")
    logger.info("\nMetrics Comparison Table:")
    print(eval_output["table"].to_string(index=False))


if __name__ == "__main__":
    main()
