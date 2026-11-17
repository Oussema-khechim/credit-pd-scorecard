#!/usr/bin/env python3
"""Génère un jeu de données SYNTHÉTIQUE pour tester que le pipeline tourne.

DÉMO TECHNIQUE UNIQUEMENT : ne jamais présenter les résultats obtenus sur ces
données comme des résultats de modélisation réelle. Utiliser un jeu de données
public réel (voir README) pour le projet.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 20000
age = rng.integers(20, 75, n)
income = np.exp(rng.normal(10.3, 0.5, n))
income[rng.random(n) < 0.08] = np.nan
utilization = np.clip(rng.beta(2, 3, n), 0, 1)
n_late = rng.poisson(0.4 + 1.5 * utilization)
emp_years = np.clip(rng.gamma(2.0, 4.0, n), 0, 40)
home = rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n, p=[0.40, 0.15, 0.43, 0.02])
year = rng.integers(2018, 2024, n)
noise = rng.normal(0, 0.5, n)

inc_f = np.where(np.isnan(income), np.nanmedian(income), income)
logit = (-4.2 + 2.0 * utilization + 0.55 * n_late - 0.02 * (age - 45)
         - 0.9 * (np.log(inc_f) - 10.3) - 0.03 * emp_years
         + np.where(home == "RENT", 0.35, 0.0) + noise)
default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame(dict(id=np.arange(n), year=year, age=age, income=income,
                       utilization=utilization, n_late=n_late,
                       employment_years=emp_years, home=home, default=default))
path = Path(__file__).resolve().parent.parent / "data" / "synthetic_demo.csv"
path.parent.mkdir(exist_ok=True)
df.to_csv(path, index=False)
print(f"{path} écrit ({n} lignes, taux de défaut = {default.mean():.3%}) -- DONNÉES SYNTHÉTIQUES")
