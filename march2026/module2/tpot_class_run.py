"""
TPOT Pipeline Runner Script
==========================

This script trains a TPOTClassifier on a dataset, evaluates it on a validation
split, logs progress, and saves results in a structured output directory.

Key Features
------------
- Validates metric against sklearn available scorers
- Configurable TPOT hyperparameters via CLI
- Splits dataset into train/validation
- Runs TPOT AutoML pipeline
- Logs progress to file and console
- Computes multiple classification metrics
- Saves model and metrics
- Creates timestamped run directories

Output Structure
----------------
tpot_results/
└── run_YYYYMMDD_HHMMSS/
    ├── run.log
    ├── metrics.txt
    ├── best_pipeline.pkl  (or tpot.pkl)

Arguments
---------
--data_path : str
--target : str
--test_size : float

--metric : str
    Must exist in sklearn.metrics.get_scorer_names()

--model_name : str
    "tpot" or "best_pipeline"

TPOT PARAMETERS:
--generations : int
--population_size : int
--cv : int
--random_state : int

Examples
--------
python script.py --data_path data.csv --target y \
    --generations 5 --population_size 10 --cv 5

Notes
-----
- Designed for classification
- Logging enables monitoring long TPOT runs
"""

import os
import argparse
import logging
from datetime import datetime

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    get_scorer_names,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from tpot import TPOTClassifier


def setup_logging(log_file):
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def validate_metric(metric_name):
    """Validate sklearn metric."""
    valid_metrics = get_scorer_names()
    if metric_name not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric_name}")


def compute_metrics(y_true, y_pred):
    """Compute multiple classification metrics."""
    results = {}
    try:
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        results["precision_weighted"] = precision_score(y_true, y_pred, average="weighted")
        results["recall_weighted"] = recall_score(y_true, y_pred, average="weighted")
        results["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        results["mcc"] = matthews_corrcoef(y_true, y_pred)
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        results["classification_report"] = classification_report(y_true, y_pred)
    except Exception as e:
        logging.warning(f"Metric computation issue: {e}")
    return results


def run_tpot_pipeline(
    df,
    target_column,
    test_size,
    metric,
    output_dir,
    model_name,
    generations,
    population_size,
    cv,
    random_state
):
    """
    Run TPOT pipeline with configurable parameters.
    """

    validate_metric(metric)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    setup_logging(os.path.join(run_dir, "run.log"))

    logging.info("Starting TPOT run")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logging.info("Data split done")

    tpot = TPOTClassifier(
        generations=generations,
        population_size=population_size,
        cv=cv,
        random_state=random_state,
        scoring=metric,
        verbosity=2
    )

    logging.info("Training TPOT...")
    tpot.fit(X_train, y_train)

    preds = tpot.predict(X_val)

    metrics = compute_metrics(y_val, preds)

    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    logging.info("Metrics saved")

    if model_name == "tpot":
        model_to_save = tpot
    elif model_name == "best_pipeline":
        model_to_save = tpot.fitted_pipeline_
    else:
        raise ValueError("model_name must be 'tpot' or 'best_pipeline'")

    model_path = os.path.join(run_dir, f"{model_name}.pkl")
    joblib.dump(model_to_save, model_path)

    logging.info(f"Model saved: {model_path}")
    logging.info("Run finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TPOT pipeline")

    # Core args
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--metric", type=str, default="accuracy")
    parser.add_argument("--output_dir", type=str, default="tpot_results")
    parser.add_argument("--model_name", type=str, default="best_pipeline")

    # TPOT args
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--cv", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    run_tpot_pipeline(
        df=df,
        target_column=args.target,
        test_size=args.test_size,
        metric=args.metric,
        output_dir=args.output_dir,
        model_name=args.model_name,
        generations=args.generations,
        population_size=args.population_size,
        cv=args.cv,
        random_state=args.random_state
    )
