"""Métriques de discrimination, calibration et stabilité."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def discrimination(y, p) -> dict:
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"auc": float(auc), "gini": float(2 * auc - 1),
            "ks": float(np.max(tpr - fpr))}


def calibration_table(y, p, n_groups: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["grp"] = pd.qcut(df["p"].rank(method="first"), n_groups, labels=False)
    t = df.groupby("grp").agg(n=("y", "size"), mean_pd=("p", "mean"),
                              observed=("y", "mean")).reset_index()
    t["ratio_obs_pred"] = t["observed"] / t["mean_pd"]
    return t


def psi_continuous(expected, actual, n_bins: int = 10, eps: float = 1e-6) -> float:
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
    pe = np.bincount(np.searchsorted(edges, expected), minlength=len(edges) + 1) / len(expected)
    pa = np.bincount(np.searchsorted(edges, actual), minlength=len(edges) + 1) / len(actual)
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_discrete(expected, actual, eps: float = 1e-6) -> float:
    e = pd.Series(np.round(np.asarray(expected, float), 10)).value_counts(normalize=True)
    a = pd.Series(np.round(np.asarray(actual, float), 10)).value_counts(normalize=True)
    idx = e.index.union(a.index)
    pe = e.reindex(idx).fillna(0).clip(lower=eps).to_numpy()
    pa = a.reindex(idx).fillna(0).clip(lower=eps).to_numpy()
    return float(np.sum((pa - pe) * np.log(pa / pe)))
