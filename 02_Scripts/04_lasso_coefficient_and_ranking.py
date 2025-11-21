#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 4: LASSO COEFFICIENTS ANALYSIS
- Carga los mejores modelos LASSO por outer split.
- Extrae coeficientes (multiclase OVR), identifica genes no-cero y ranking por magnitud.
- Exporta:
    * lasso_nonzero_by_outer.csv (outer_id, gene, max_abs_coef, sign_by_class)
    * lasso_top_by_class_outer{j}.csv (por clase)
"""

import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("results/best_models")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

def outer_ids_in_dir(pattern=r"lasso_outer_(\d+)\.joblib"):
    ids = []
    for p in MODELS_DIR.glob("lasso_outer_*.joblib"):
        m = re.search(pattern, p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)

def main():
    rows = []
    for oid in outer_ids_in_dir():
        model = joblib.load(MODELS_DIR / f"lasso_outer_{oid}.joblib")
        clf = model.named_steps["clf"]
        scaler = model.named_steps["scaler"]
        coef = clf.coef_  # shape: (n_classes, n_features)
        absmax = np.max(np.abs(coef), axis=0)
        nonzero_mask = absmax != 0
        # nombres de features
        # OJO: StandardScaler no cambia nombres. Tomamos del entrenamiento guardado:
        X_train = pd.read_csv("01_Data/processed/X_train.csv", index_col=0)
        genes = X_train.columns.tolist()

        for j, g in enumerate(genes):
            if nonzero_mask[j]:
                sign_by_class = np.sign(coef[:, j]).astype(int).tolist()
                rows.append({
                    "outer_id": oid,
                    "gene": g,
                    "max_abs_coef": absmax[j],
                    "sign_by_class": sign_by_class
                })

        # ranking por clase
        for c in range(coef.shape[0]):
            order = np.argsort(-np.abs(coef[c, :]))
            dfc = pd.DataFrame({
                "gene": [genes[idx] for idx in order],
                "coef": coef[c, order],
                "abs_coef": np.abs(coef[c, order])
            })
            dfc.to_csv(OUT_DIR / f"lasso_top_by_class_outer{oid}_class{c}.csv", index=False)

    pd.DataFrame(rows).sort_values(["outer_id", "max_abs_coef"], ascending=[True, False])\
        .to_csv(OUT_DIR / "lasso_nonzero_by_outer.csv", index=False)

    print("OK - Coeficientes LASSO exportados.")

if __name__ == "__main__":
    main()
