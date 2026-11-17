#!/usr/bin/env python3
"""Pipeline PD : binning WoE -> sélection -> régression logistique -> scorecard
-> évaluation (discrimination, calibration, stabilité, backtesting).

Exemple :
    python src/run_pd_model.py --data data/credit.csv --target default \
        --id-col id --time-col year --out outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

import metrics as M
from binning import WoEBinner


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV d'entrée")
    ap.add_argument("--target", required=True, help="colonne cible (1 = défaut)")
    ap.add_argument("--id-col", default=None, help="colonne identifiant à exclure")
    ap.add_argument("--time-col", default=None,
                    help="colonne temporelle : active le split out-of-time")
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--min-iv", type=float, default=0.02)
    ap.add_argument("--max-corr", type=float, default=0.7)
    ap.add_argument("--max-bins", type=int, default=10)
    ap.add_argument("--pdo", type=float, default=20.0)
    ap.add_argument("--base-score", type=float, default=600.0)
    ap.add_argument("--base-odds", type=float, default=50.0, help="odds good:bad au score de base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.data)
    if a.target not in df.columns:
        sys.exit(f"Colonne cible '{a.target}' introuvable.")
    y = df[a.target]
    if not set(y.dropna().unique()) <= {0, 1}:
        sys.exit("La cible doit être binaire (0/1).")
    y = y.astype(int)
    excluded = {a.target, a.id_col, a.time_col}
    X = df[[c for c in df.columns if c not in excluded]]

    # ---- split : out-of-time si une colonne temporelle est fournie
    if a.time_col:
        order = np.argsort(df[a.time_col].to_numpy(), kind="stable")
        cut = int(len(df) * (1 - a.test_size))
        tr_idx, te_idx = order[:cut], order[cut:]
    else:
        tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=a.test_size,
                                          stratify=y, random_state=a.seed)
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]

    # ---- binning WoE + IV
    binner = WoEBinner(max_bins=a.max_bins).fit(Xtr, ytr)
    iv = binner.iv_
    iv.rename("iv").to_csv(out / "iv_table.csv")
    suspect = iv[iv > 0.5]
    if len(suspect):
        print(f"[ATTENTION] IV > 0.5 (fuite d'information possible ?) : {list(suspect.index)}")

    # ---- sélection : IV minimal puis élimination des variables corrélées
    cand = iv[iv >= a.min_iv].index.tolist()
    Wtr_all = binner.transform(Xtr, cand)
    kept: list[str] = []
    for c in cand:  # déjà triées par IV décroissant
        if all(abs(Wtr_all[c].corr(Wtr_all[k])) < a.max_corr for k in kept):
            kept.append(c)
    if not kept:
        sys.exit("Aucune variable retenue (baisser --min-iv ?).")

    # ---- régression logistique ; on retire toute variable au coefficient <= 0
    #      (WoE croissant avec le risque => signe attendu positif)
    cols = kept.copy()
    while True:
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        model.fit(Wtr_all[cols], ytr)
        coefs = pd.Series(model.coef_[0], index=cols)
        neg = coefs[coefs <= 0]
        if neg.empty or len(cols) == 1:
            break
        cols.remove(neg.idxmin())
    coefs.rename("coef").to_csv(out / "coefficients.csv")
    intercept = float(model.intercept_[0])

    Wtr = binner.transform(Xtr, cols)
    Wte = binner.transform(Xte, cols)
    p_tr = model.predict_proba(Wtr)[:, 1]
    p_te = model.predict_proba(Wte)[:, 1]

    # ---- scorecard : score = offset - factor * logit(PD)
    factor = a.pdo / np.log(2)
    offset = a.base_score - factor * np.log(a.base_odds)
    m = len(cols)
    rows = []
    for c in cols:
        sp = binner.specs_[c]
        for lab, n, w in zip(sp["labels"], sp["n"], sp["woe"]):
            pts = -(coefs[c] * w + intercept / m) * factor + offset / m
            rows.append(dict(variable=c, bin=lab, n=int(n), woe=w, points=round(pts, 1)))
    pd.DataFrame(rows).to_csv(out / "scorecard_points.csv", index=False)
    binner.bin_table(cols).to_csv(out / "binning_tables.csv", index=False)
    score_tr = offset - factor * np.log(p_tr / (1 - p_tr))
    score_te = offset - factor * np.log(p_te / (1 - p_te))

    # ---- évaluation
    res = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "default_rate_train": float(ytr.mean()), "default_rate_test": float(yte.mean()),
        "split": "out-of-time" if a.time_col else "aléatoire stratifié",
        "variables": cols,
        "train": M.discrimination(ytr, p_tr),
        "test": M.discrimination(yte, p_te),
        "mean_pd_test": float(p_te.mean()),
        "psi_score_train_vs_test": M.psi_continuous(score_tr, score_te),
        "score_range_test": [float(score_te.min()), float(score_te.max())],
    }
    psi_vars = {c: M.psi_discrete(Wtr[c], Wte[c]) for c in cols}
    res["psi_variables"] = psi_vars
    calib = M.calibration_table(yte, p_te)
    calib.to_csv(out / "calibration_deciles.csv", index=False)

    if a.time_col:
        te_time = df[a.time_col].iloc[te_idx].reset_index(drop=True)
        if te_time.nunique() > 12:
            te_time = pd.qcut(te_time.rank(method="first"), 6, labels=False)
        bt = pd.DataFrame({"period": te_time, "y": yte.to_numpy(), "p": p_te})
        def _agg(g):
            auc = roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan
            return pd.Series({"n": len(g), "observed": g["y"].mean(),
                              "mean_pd": g["p"].mean(), "auc": auc})
        bt.groupby("period").apply(_agg).reset_index().to_csv(out / "backtest_by_period.csv", index=False)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # ---- graphiques
    fig, ax = plt.subplots(figsize=(5, 5))
    for lab, yy, pp in (("train", ytr, p_tr), ("test", yte, p_te)):
        fpr, tpr, _ = roc_curve(yy, pp)
        ax.plot(fpr, tpr, label=f"{lab} (Gini = {res[lab]['gini']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbe ROC")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(calib["mean_pd"], calib["observed"], "o-", label="déciles de PD (test)")
    lim = max(calib["mean_pd"].max(), calib["observed"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="calibration parfaite")
    ax.set(xlabel="PD moyenne prédite", ylabel="Taux de défaut observé", title="Calibration")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=150); plt.close(fig)

    # ---- résumé console
    print(f"Variables retenues ({len(cols)}) : {cols}")
    print(f"Gini train / test : {res['train']['gini']:.3f} / {res['test']['gini']:.3f}")
    print(f"KS   train / test : {res['train']['ks']:.3f} / {res['test']['ks']:.3f}")
    print(f"PD moyenne test = {res['mean_pd_test']:.4f} vs taux observé = {res['default_rate_test']:.4f}")
    print(f"PSI score (train vs test) = {res['psi_score_train_vs_test']:.4f}")
    print(f"Sorties écrites dans {out}/")


if __name__ == "__main__":
    main()
