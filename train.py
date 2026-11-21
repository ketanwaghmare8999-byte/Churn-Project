"""Train, tune, evaluate and persist the best churn model.

Usage:
    python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline

from src.features import RAW_FEATURES, TARGET, build_preprocessor, load_and_clean

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

SEED = 42
MODEL_PATH = Path("models/churn_model.joblib")
METADATA_PATH = Path("models/metadata.json")


@dataclass(frozen=True)
class BusinessCase:
    """Economics of a retention campaign (all values per customer)."""
    offer_cost: float = 20.0      # cost of the incentive, paid for every flagged customer
    customer_value: float = 500.0  # revenue kept if a churner is retained
    success_rate: float = 0.30     # share of contacted churners who accept the offer

    def net_benefit(self, y_true: np.ndarray, flagged: np.ndarray) -> float:
        """Total campaign profit vs. doing nothing (which scores 0)."""
        tp = np.sum(flagged & (y_true == 1))
        return float(tp * self.success_rate * self.customer_value - flagged.sum() * self.offer_cost)


def candidate_searches(cv: StratifiedKFold) -> dict[str, object]:
    """Each model gets a hyper-parameter search, scored on AUC-ROC with stratified CV."""
    def make(model, params, randomized=False, n_iter=12):
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", model)])
        kwargs = dict(scoring="roc_auc", cv=cv, n_jobs=-1)
        if randomized:
            return RandomizedSearchCV(pipe, params, n_iter=n_iter, random_state=SEED, **kwargs)
        return GridSearchCV(pipe, params, **kwargs)

    return {
        "logistic_regression": make(
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": make(
            RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [6, 10, None],
             "clf__min_samples_leaf": [1, 5, 10], "clf__max_features": ["sqrt", 0.5]},
            randomized=True,
        ),
        "gradient_boosting": make(
            HistGradientBoostingClassifier(random_state=SEED),
            {"clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [3, 5, None],
             "clf__max_iter": [100, 200, 300], "clf__l2_regularization": [0.0, 1.0, 10.0]},
            randomized=True,
        ),
    }


def best_threshold(y_true: np.ndarray, proba: np.ndarray, case: BusinessCase) -> float:
    """Probability cut-off that maximises campaign profit."""
    grid = np.linspace(0.05, 0.95, 91)
    profits = [case.net_benefit(y_true, proba >= t) for t in grid]
    return round(float(grid[int(np.argmax(profits))]), 2)


def evaluate(y_true, proba, threshold: float, case: BusinessCase) -> dict[str, float]:
    pred = proba >= threshold
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
        "net_benefit": case.net_benefit(np.asarray(y_true), pred),
        "net_benefit_per_customer": case.net_benefit(np.asarray(y_true), pred) / len(y_true),
    }


def main(data_path: str, case: BusinessCase) -> None:
    df = load_and_clean(data_path)
    log.info("Loaded %d customers (churn rate %.1f%%)", len(df), 100 * df[TARGET].mean())

    X_train, X_test, y_train, y_test = train_test_split(
        df[RAW_FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, search in candidate_searches(cv).items():
        log.info("Tuning %s ...", name)
        search.fit(X_train, y_train)
        # Out-of-fold probabilities -> pick the profit-maximising threshold WITHOUT touching the test set
        oof = cross_val_predict(search.best_estimator_, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
        threshold = best_threshold(y_train.to_numpy(), oof, case)
        cv_profit = case.net_benefit(y_train.to_numpy(), oof >= threshold) / len(y_train)
        test_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
        results[name] = {
            "best_params": search.best_params_,
            "cv_roc_auc": search.best_score_,
            "cv_net_benefit_per_customer": cv_profit,
            "test": evaluate(y_test, test_proba, threshold, case),
        }
        fitted[name] = search.best_estimator_
        log.info("%s | CV AUC %.3f | CV profit/customer $%.2f | test AUC %.3f | test PR-AUC %.3f",
                 name, search.best_score_, cv_profit,
                 results[name]["test"]["roc_auc"], results[name]["test"]["pr_auc"])

    # Model selection uses cross-validated business impact, never the held-out test set
    winner = max(results, key=lambda m: results[m]["cv_net_benefit_per_customer"])
    log.info("Selected model: %s", winner)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(fitted[winner], MODEL_PATH)
    METADATA_PATH.write_text(json.dumps({
        "model_name": winner,
        "threshold": results[winner]["test"]["threshold"],
        "business_case": asdict(case),
        "results": results,
    }, indent=2, default=float))

    summary = pd.DataFrame({m: r["test"] for m, r in results.items()}).T.round(3)
    print("\nHeld-out test results\n", summary.to_string())
    print(f"\nSaved {winner} -> {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--offer-cost", type=float, default=BusinessCase.offer_cost)
    parser.add_argument("--customer-value", type=float, default=BusinessCase.customer_value)
    parser.add_argument("--success-rate", type=float, default=BusinessCase.success_rate)
    args = parser.parse_args()
    main(args.data, BusinessCase(args.offer_cost, args.customer_value, args.success_rate))
