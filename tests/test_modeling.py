import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pharma_adherence.modeling import ModelTrainer

"""
RUN THIS SCRIPT USING `PYTHONPATH=src pytest -v`
"""

def make_linear_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "days_supply": [30, 60, 30, 90, 60, 30, 90, 60, 30, 90],
            "sex": ["female", "male", "female", "male", "female", "male", "female", "male", "female", "male"],
            "pharmacy_name": ["walgreens", "cvs", "walgreens", "cvs", "walgreens", "cvs", "walgreens", "cvs", "walgreens", "cvs"],
            "copay_amount": [10.0, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 16.0],
        }
    )


def make_logistic_df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "days_supply": [30, 60, 30, 90, 60, 30, 90, 60, 30, 90],
            "sex": ["female", "male", "female", "male", "female", "male", "female", "male", "female", "male"],
            "pharmacy_name": ["walgreens", "cvs", "walgreens", "cvs", "walgreens", "cvs", "walgreens", "cvs", "walgreens", "cvs"],
            "adherent": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_build_preprocessor_detects_numeric_and_categorical():
    df = make_linear_df()
    trainer = ModelTrainer(df=df, target="copay_amount", features=["age", "days_supply", "sex", "pharmacy_name"])

    X = df[trainer.features]
    preprocessor = trainer.build_preprocessor(X)

    assert preprocessor.__class__.__name__ == "ColumnTransformer"
    assert preprocessor.transformers[0][0] == "num"
    assert preprocessor.transformers[1][0] == "cat"
    assert set(preprocessor.transformers[0][2]) == {"age", "days_supply"}
    assert set(preprocessor.transformers[1][2]) == {"sex", "pharmacy_name"}


def test_train_linear_returns_model_and_metrics():
    df = make_linear_df()
    trainer = ModelTrainer(
        df=df,
        target="copay_amount",
        features=["age", "days_supply", "sex", "pharmacy_name"],
    )

    model, metrics = trainer.train_linear()

    assert isinstance(model, Pipeline)
    assert trainer.model is model
    assert trainer.metrics == metrics

    assert set(metrics.keys()) == {"mse", "mae", "r2"}
    assert np.isfinite(metrics["mse"])
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["r2"])

    assert metrics["mse"] >= 0


def test_train_logistic_returns_model_and_metrics():
    df = make_logistic_df()
    trainer = ModelTrainer(
        df=df,
        target="adherent",
        features=["age", "days_supply", "sex", "pharmacy_name"],
    )

    model, metrics = trainer.train_logistic()

    assert isinstance(model, Pipeline)
    assert trainer.model is model
    assert trainer.metrics == metrics

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "roc_auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_raises_before_training():
    df = make_linear_df()
    trainer = ModelTrainer(
        df=df,
        target="copay_amount",
        features=["age", "days_supply", "sex", "pharmacy_name"],
    )

    with pytest.raises(ValueError, match="No metrics available yet"):
        trainer.evaluate()


def test_evaluate_returns_metrics_after_training():
    df = make_logistic_df()
    trainer = ModelTrainer(
        df=df,
        target="adherent",
        features=["age", "days_supply", "sex", "pharmacy_name"],
    )

    _, metrics = trainer.train_logistic()
    assert trainer.evaluate() == metrics