"""Model evaluation: discrimination, ranking, calibration and business-cost threshold selection.

Positive class = churn (no purchase in the next 30 days). Accuracy is deliberately not reported as a headline
number: with ~58% positives a model that predicts "everyone churns" scores 58% accuracy.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def top_k_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the highest-scoring `fraction` of rows (ties broken by position, deterministic)."""
    n = len(scores)
    k = max(1, int(round(n * fraction)))
    order = np.argsort(-np.asarray(scores), kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def recall_at_top(y: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    """Share of ALL churners that fall in the highest-risk `fraction` of customers."""
    y = np.asarray(y)
    return float(y[top_k_mask(scores, fraction)].sum() / y.sum()) if y.sum() else float("nan")


def precision_at_top(y: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    y = np.asarray(y)
    return float(y[top_k_mask(scores, fraction)].mean())


def lift_at_top(y: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    """Churn rate in the top group divided by the overall churn rate."""
    return precision_at_top(y, scores, fraction) / float(np.mean(y))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Reliability table over equal-count probability bins."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")).reset_index()
    return g


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = calibration_table(y, p, bins)
    return float((t["n"] * (t["mean_predicted"] - t["observed_rate"]).abs()).sum() / t["n"].sum())


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(pred)),     # reported for completeness, never as evidence of success
        "flagged_share": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    """Threshold-free metrics (valid for any score where higher = more likely to churn)."""
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "prevalence": float(np.mean(y)),
        "recall_at_top10pct": recall_at_top(y, scores, 0.10),
        "precision_at_top10pct": precision_at_top(y, scores, 0.10),
        "lift_at_top10pct": lift_at_top(y, scores, 0.10),
        "recall_at_top30pct": recall_at_top(y, scores, 0.30),
        "precision_at_top30pct": precision_at_top(y, scores, 0.30),
        "n": int(len(y)),
    }


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"brier": float(brier_score_loss(y, p)), "expected_calibration_error": expected_calibration_error(y, p),
            "mean_predicted": float(np.mean(p)), "observed_rate": float(np.mean(y))}


# ------------------------------- business-cost threshold -------------------------------
def expected_value_per_contact(p: np.ndarray, biz: Mapping[str, float]) -> np.ndarray:
    """Expected $ of contacting a customer whose churn probability is p.

    p * save_rate * value_if_saved           a churner is saved and keeps buying
    - (1-p) * redeem_rate * discount_cost    a non-churner takes the discount for a purchase they would have made
    - contact_cost                           everyone contacted costs something
    """
    p = np.asarray(p, dtype=float)
    return p * biz["save_rate"] * biz["value_if_saved"] - (1 - p) * biz["redeem_rate"] * biz["discount_cost"] - biz["contact_cost"]


def campaign_value(y: np.ndarray, p: np.ndarray, threshold: float, biz: Mapping[str, float]) -> float:
    """Realised $ of contacting everyone with p >= threshold, using the ACTUAL outcomes y."""
    y = np.asarray(y)
    contact = np.asarray(p) >= threshold
    gain = (y[contact] == 1).sum() * biz["save_rate"] * biz["value_if_saved"]
    loss = (y[contact] == 0).sum() * biz["redeem_rate"] * biz["discount_cost"] + contact.sum() * biz["contact_cost"]
    return float(gain - loss)


def choose_threshold(y: np.ndarray, p: np.ndarray, biz: Mapping[str, float], grid: int = 200) -> dict:
    """Threshold maximising realised campaign value on the VALIDATION set (never on test)."""
    ts = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    values = np.array([campaign_value(y, p, t, biz) for t in ts])
    best = int(np.argmax(values))
    return {"threshold": float(ts[best]), "validation_value": float(values[best]),
            "contact_all_value": campaign_value(y, p, -np.inf, biz), "contact_none_value": 0.0,
            "breakeven_probability": float((biz["contact_cost"] + biz["redeem_rate"] * biz["discount_cost"]) /
                                            (biz["save_rate"] * biz["value_if_saved"] + biz["redeem_rate"] * biz["discount_cost"]))}
