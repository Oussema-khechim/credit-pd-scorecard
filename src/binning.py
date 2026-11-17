"""Binning WoE (monotone pour les variables numériques) et Information Value.

Convention : WoE = ln(%bad / %good). Un WoE élevé correspond donc à un risque
élevé, et les coefficients de la régression logistique sur les WoE doivent
être strictement positifs (contrôle de cohérence utilisé dans le pipeline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _counts(codes: np.ndarray, y: np.ndarray, n_bins: int):
    n = np.bincount(codes, minlength=n_bins).astype(float)
    bad = np.bincount(codes, weights=y, minlength=n_bins).astype(float)
    return n, bad


class WoEBinner:
    def __init__(self, max_bins: int = 10, min_bin_share: float = 0.05,
                 smoothing: float = 0.5):
        self.max_bins = max_bins
        self.min_bin_share = min_bin_share
        self.smoothing = smoothing

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        y_arr = np.asarray(y, dtype=float)
        self.total_bad_ = float(y_arr.sum())
        self.total_good_ = float(len(y_arr) - y_arr.sum())
        self.specs_: dict[str, dict] = {}
        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                self.specs_[col] = self._fit_numeric(s, y_arr)
            else:
                self.specs_[col] = self._fit_categorical(s.astype("object"), y_arr)
        self.iv_ = pd.Series({c: sp["iv"] for c, sp in self.specs_.items()}
                             ).sort_values(ascending=False)
        return self

    def _woe_iv(self, n: np.ndarray, bad: np.ndarray):
        good = n - bad
        eps, k = self.smoothing, len(n)
        pb = (bad + eps) / (self.total_bad_ + eps * k)
        pg = (good + eps) / (self.total_good_ + eps * k)
        woe = np.log(pb / pg)
        iv = float(np.sum((pb - pg) * woe))
        return woe, iv

    @staticmethod
    def _codes(x: np.ndarray, edges: list[float]) -> np.ndarray:
        return np.asarray(pd.cut(x, bins=edges, labels=False), dtype=int)

    def _merge(self, x, yy, edges, direction):
        """Fusionne les classes adjacentes jusqu'à obtenir la monotonie du
        taux de défaut et une taille minimale par classe."""
        min_n = self.min_bin_share * len(x)
        while len(edges) > 2:
            codes = self._codes(x, edges)
            n, bad = _counts(codes, yy, len(edges) - 1)
            rate = bad / np.maximum(n, 1)
            idx = None
            if direction != 0:
                viol = np.where(np.diff(rate) * direction < 0)[0]
                if len(viol):
                    idx = int(viol[0])
            if idx is None:
                small = np.where(n < min_n)[0]
                if len(small) == 0:
                    break
                i = int(small[np.argmin(n[small])])
                if i == 0:
                    idx = 0
                elif i == len(n) - 1:
                    idx = i - 1
                else:
                    idx = i - 1 if n[i - 1] <= n[i + 1] else i
            edges.pop(idx + 1)
        return edges

    def _fit_numeric(self, s: pd.Series, y: np.ndarray) -> dict:
        arr = s.to_numpy(dtype=float)
        mask = ~np.isnan(arr)
        x, yy = arr[mask], y[mask]
        edges = [-np.inf, np.inf]
        if len(np.unique(x)) > 1:
            qs = np.unique(np.quantile(x, np.linspace(0, 1, self.max_bins + 1)[1:-1],
                                       method="lower"))
            qs = qs[qs < x.max()]
            edges = [-np.inf, *qs.tolist(), np.inf]
            rho = spearmanr(x, yy).statistic
            direction = 0 if np.isnan(rho) else (1 if rho >= 0 else -1)
            edges = self._merge(x, yy, edges, direction)
        codes = self._codes(x, edges)
        nb = len(edges) - 1
        n, bad = _counts(codes, yy, nb)
        labels = [f"({edges[i]:.4g}, {edges[i + 1]:.4g}]" for i in range(nb)]
        na_n = int((~mask).sum())
        if na_n > 0:
            n = np.append(n, na_n)
            bad = np.append(bad, y[~mask].sum())
            labels.append("NA")
        woe, iv = self._woe_iv(n, bad)
        return dict(type="num", edges=edges, nb=nb, has_na=na_n > 0,
                    woe=woe, n=n, bad=bad, labels=labels, iv=iv)

    def _fit_categorical(self, s: pd.Series, y: np.ndarray) -> dict:
        s = s.fillna("NA").astype(str)
        share = s.value_counts(normalize=True)
        rare = set(share[share < self.min_bin_share].index)
        mapped = s.where(~s.isin(rare), "OTHER")
        cats = sorted(mapped.unique())
        codes = pd.Categorical(mapped, categories=cats).codes
        n, bad = _counts(np.asarray(codes, dtype=int), y, len(cats))
        woe, iv = self._woe_iv(n, bad)
        return dict(type="cat", cats=cats, woe=woe, n=n, bad=bad,
                    labels=list(cats), iv=iv)

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
        cols = cols or list(self.specs_)
        out = {}
        for c in cols:
            sp = self.specs_[c]
            if sp["type"] == "num":
                arr = X[c].to_numpy(dtype=float)
                na = np.isnan(arr)
                woe_bins = sp["woe"][: sp["nb"]]
                na_woe = sp["woe"][sp["nb"]] if sp["has_na"] else 0.0
                codes = np.asarray(pd.cut(np.where(na, 0.0, arr), bins=sp["edges"],
                                          labels=False), dtype=int)
                out[c] = np.where(na, na_woe, woe_bins[codes])
            else:
                lookup = dict(zip(sp["cats"], sp["woe"]))
                s = X[c].astype("object").fillna("NA").astype(str)
                m = s.map(lambda v: v if v in lookup else "OTHER")
                out[c] = m.map(lookup).fillna(0.0).to_numpy(dtype=float)
        return pd.DataFrame(out, index=X.index)

    # --------------------------------------------------------------- tables
    def bin_table(self, cols: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for c in (cols or list(self.specs_)):
            sp = self.specs_[c]
            total = sp["n"].sum()
            for lab, n, b, w in zip(sp["labels"], sp["n"], sp["bad"], sp["woe"]):
                rows.append(dict(variable=c, bin=lab, n=int(n), share=n / total,
                                 bad_rate=b / n if n else np.nan, woe=w))
        return pd.DataFrame(rows)
