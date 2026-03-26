"""
TPOT Pipeline Runner Script
==========================

This script trains a TPOTClassifier on a dataset, evaluates it on a validation
split, logs progress, and saves results in a structured output directory.

Key Features
------------
- Validates metric against sklearn available scorers
- Splits dataset into train/validation
- Runs TPOT AutoML pipeline
- Logs progress to file and console
- Computes multiple classification metrics
- Saves model and metrics
- Creates timestamped run directories

Output Structure
----------------
All outputs are saved inside `output_dir` (default: tpot_results):

tpot_results/
└── run_YYYYMMDD_HHMMSS/
    ├── run.log                # Full execution log
    ├── metrics.txt            # All computed validation metrics
    ├── best_pipeline.pkl      # Saved model (or tpot_model.pkl)

Arguments
---------
--data_path : str
    Path to input CSV file

--target : str
    Target column name

--test_size : float (default=0.2)
    Fraction of data used for validation

--metric : str (default="accuracy")
    Must be a valid sklearn scorer (checked before execution)

--output_dir : str (default="tpot_results")
    Base directory for outputs

--model_name : str (default="best_pipeline")
    Which model to save:
        - "tpot" → full TPOT object
        - "best_pipeline" → fitted sklearn pipeline

Examples
--------
Basic usage:
    python script.py --data_path data.csv --target label

Custom metric and split:
    python script.py --data_path data.csv --target label \
                     --metric f1 --test_size 0.3

Save full TPOT object:
    python script.py --data_path data.csv --target label \
                     --model_name tpot

Notes
-----
- Metric must exist in sklearn.metrics.get_scorer_names()
- Designed for classification tasks
- Logging file allows independent monitoring of progress
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
    """
    Configure logging to file and console.

    Parameters
    ----------
    log_file : str
        Path to log file

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def validate_metric(metric_name):
    """
    Validate that the provided metric exists in sklearn.

    Parameters
    ----------
    metric_name : str
        Name of the metric to validate

    Raises
    ------
    ValueError
        If metric is not found in sklearn scorers
    """
    valid_metrics = get_scorer_names()
    if metric_name not in valid_metrics:
        raise ValueError(
            f"Metric '{metric_name}' not found.\n"
            f"Available metrics: {list(valid_metrics)}"
        )


def compute_metrics(y_true, y_pred):
    """
    Compute a wide set of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels

    y_pred : array-like
        Predicted labels

    Returns
    -------
    dict
        Dictionary containing computed metrics
    """
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
    test_size=0.2,
    metric="accuracy",
    output_dir="tpot_results",
    model_name="best_pipeline"
):
    """
    Execute TPOT pipeline training, evaluation, and saving.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset

    target_column : str
        Column to be used as target

    test_size : float, optional
        Validation split ratio (default=0.2)

    metric : str, optional
        Scoring metric (must exist in sklearn)

    output_dir : str, optional
        Base output directory

    model_name : str, optional
        Model to save:
            - "tpot"
            - "best_pipeline"

    Returns
    -------
    None
    """

    # Validate metric
    validate_metric(metric)

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Logging
    log_file = os.path.join(run_dir, "run.log")
    setup_logging(log_file)

    logging.info("Starting TPOT pipeline")

    # Split data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    logging.info("Data split completed")

    # Initialize TPOT
    tpot = TPOTClassifier(
        random_state=42,
        cv=2,
        generations=2,
        population_size=2,
        scoring=metric,
        verbosity=2
    )

    logging.info("Training TPOT model...")
    tpot.fit(X_train, y_train)

    logging.info("Training completed")

    # Predict
    preds = tpot.predict(X_val)

    # Metrics
    metrics = compute_metrics(y_val, preds)

    # Save metrics
    metrics_file = os.path.join(run_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    logging.info("Metrics saved")

    # Select model
    if model_name == "tpot":
        model_to_save = tpot
    elif model_name == "best_pipeline":
        model_to_save = tpot.fitted_pipeline_
    else:
        raise ValueError("model_name must be 'tpot' or 'best_pipeline'")

    # Save model
    model_path = os.path.join(run_dir, f"{model_name}.pkl")
    joblib.dump(model_to_save, model_path)

    logging.info(f"Model saved to {model_path}")
    logging.info("Run completed successfully")


if __name__ == "__main__":
    """
    Command-line interface entry point.
    Parses arguments and executes the pipeline.
    """

    parser = argparse.ArgumentParser(description="Run TPOT pipeline")

    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True,
                        help="Target column name")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Validation split size")
    parser.add_argument("--metric", type=str, default="accuracy",
                        help="Sklearn metric name")
    parser.add_argument("--output_dir", type=str, default="tpot_results",
                        help="Output directory")
    parser.add_argument("--model_name", type=str, default="best_pipeline",
                        help="'tpot' or 'best_pipeline'")

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    run_tpot_pipeline(
        df=df,
        target_column=args.target,
        test_size=args.test_size,
        metric=args.metric,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
