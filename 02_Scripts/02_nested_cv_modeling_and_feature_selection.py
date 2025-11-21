#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# -----------------------------------------------------------------------------
# TFM_BreastCancer_HistologicGrade
# Script 2: 2_model_training_and_selection.py
# -----------------------------------------------------------------------------
#
# Descripción: MODEL TRAINING & SELECTION (Nested CV con splits persistentes)
# 1. Carga X_train, y_train y outer_splits.json.
# 2. Compara LASSO vs SVM-RFE con GridSearch interno (inner CV).
# 3. Incluye SimpleImputer(mean) en ambos pipelines para manejar los NaNs.
# 4. Registra balanced_accuracy, tamaño de firma, tiempos y mejores hiperparámetros.
# 5. Guarda firmas por outer split en JSON por método.
# -----------------------------------------------------------------------------
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# --- Importaciones de SciKit-Learn ---
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.impute import SimpleImputer # <<-- ¡AÑADIDO!

# ----------------------------
# 0. CONFIG
# ----------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE = Path("01_Data/processed")
X_TRAIN_PATH = BASE / "X_train.csv"
Y_TRAIN_PATH = BASE / "y_train.csv"
SPLITS_PATH = BASE / "outer_splits.json"
HOLDOUT_PATH = BASE / "test_set_20_percent.joblib"

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True, parents=True)

RESULTS_CSV = OUT_DIR / "results_nested_cv.csv"
SIG_LASSO_JSON = OUT_DIR / "signatures_LASSO.json"
SIG_RFE_JSON = OUT_DIR / "signatures_SVMRFE.json"
BEST_MODELS_DIR = OUT_DIR / "best_models"
BEST_MODELS_DIR.mkdir(exist_ok=True)

# Inner CV
INNER_SPLITS = 5
INNER_RANDOM_STATE = 1337

# ----------------------------
# 1. Carga datos y splits
# ----------------------------
print(f"Cargando datos de entrenamiento desde {X_TRAIN_PATH}...")
X_train = pd.read_csv(X_TRAIN_PATH, index_col=0)
y_train = pd.read_csv(Y_TRAIN_PATH)["Grade"].values

with open(SPLITS_PATH) as f:
    outer = json.load(f)["splits"]

n_features_total = X_train.shape[1]
feature_names = X_train.columns.tolist()

print(f"Datos cargados: X_train {X_train.shape} | Total de splits externos: {len(outer)}")

# ----------------------------
# 2. Definición de pipelines + grids (Imputer añadido)
# ----------------------------

# LASSO (Logistic Regression L1)
pipe_lasso = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")), # <--- FIX AÑADIDO: Maneja NaNs
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("clf", LogisticRegression(penalty="l1", solver="liblinear", multi_class="ovr",
                               class_weight="balanced", max_iter=5000, random_state=RANDOM_SEED))
])

grid_lasso = {
    "clf__C": np.logspace(-3, 3, 13)  # 1e-3 ... 1e3
}

# SVM-RFE: filtro univariante + RFE + clasificador lineal
pipe_rfe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")), # <--- FIX AÑADIDO: Maneja NaNs
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("selector", SelectKBest(score_func=f_classif, k=5000)),  # filtra ruido fuerte
    ("rfe", RFE(estimator=LinearSVC(C=1.0, penalty="l2", dual=True, max_iter=5000, random_state=RANDOM_SEED),
                step=0.2)),
    ("svm", SVC(kernel="linear", class_weight="balanced", probability=True, random_state=RANDOM_SEED))
])

grid_rfe = {
    "selector__k": [1000, 2000, 3000, 5000],
    "rfe__n_features_to_select": [10, 20, 30, 50, 100],
    "svm__C": np.logspace(-3, 2, 10)
}

inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_RANDOM_STATE)

def make_search(pipe, grid):
    return GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

# ----------------------------
# 3. Nested CV loop (usando splits persistentes)
# ----------------------------
rows = []
signatures_lasso = []
signatures_rfe = []

for i, split in enumerate(outer, start=1):
    print(f"\n--- Outer Fold {i}/{len(outer)} ---")
    tr_idx = split["train"]
    te_idx = split["test"]

    X_tr, X_te = X_train.iloc[tr_idx], X_train.iloc[te_idx]
    y_tr, y_te = y_train[tr_idx], y_train[te_idx]

    # ----- LASSO -----
    print("  -> Fitting LASSO...")
    gs_lasso = make_search(pipe_lasso, grid_lasso)
    t0 = time.perf_counter()
    gs_lasso.fit(X_tr, y_tr)
    t_lasso = time.perf_counter() - t0

    y_pred = gs_lasso.predict(X_te)
    bacc = balanced_accuracy_score(y_te, y_pred)
    print(f"  LASSO BACC (Outer): {bacc:.4f} | Best C: {gs_lasso.best_params_['clf__C']:.4f}")

    # Extraer firma LASSO
    best_lasso = gs_lasso.best_estimator_
    clf = best_lasso.named_steps["clf"]
    coef = clf.coef_
    mask_nonzero = np.any(coef != 0, axis=0)
    lasso_genes = [feature_names[j] for j, m in enumerate(mask_nonzero) if m]
    signatures_lasso.append(lasso_genes)

    rows.append({
        "method": "LASSO",
        "outer_id": i,
        "bACC": bacc,
        "n_features_selected": int(mask_nonzero.sum()),
        "best_params": json.dumps(gs_lasso.best_params_),
        "fit_time_sec": round(t_lasso, 3)
    })
    joblib.dump(best_lasso, BEST_MODELS_DIR / f"lasso_outer_{i}.joblib")

    # ----- SVM-RFE -----
    print("  -> Fitting SVM-RFE...")
    gs_rfe = make_search(pipe_rfe, grid_rfe)
    t0 = time.perf_counter()
    gs_rfe.fit(X_tr, y_tr)
    t_rfe = time.perf_counter() - t0

    y_pred = gs_rfe.predict(X_te)
    bacc = balanced_accuracy_score(y_te, y_pred)
    print(f"  SVM-RFE BACC (Outer): {bacc:.4f} | Best k/n_feat: {gs_rfe.best_params_['selector__k']}/{gs_rfe.best_params_['rfe__n_features_to_select']}")

    best_rfe = gs_rfe.best_estimator_
    # Obtener máscara final de RFE
    rfe = best_rfe.named_steps["rfe"]
    kbest = best_rfe.named_steps["selector"]
    selected_idx_kbest = kbest.get_support(indices=True)
    idx_selected_rfe = selected_idx_kbest[rfe.support_]
    rfe_genes = [feature_names[j] for j in idx_selected_rfe]
    signatures_rfe.append(rfe_genes)

    rows.append({
        "method": "SVM-RFE",
        "outer_id": i,
        "bACC": bacc,
        "n_features_selected": int(len(rfe_genes)),
        "best_params": json.dumps(gs_rfe.best_params_),
        "fit_time_sec": round(t_rfe, 3)
    })
    joblib.dump(best_rfe, BEST_MODELS_DIR / f"svmrfe_outer_{i}.joblib")


# ----------------------------
# 4. Guardar resultados y firmas
# ----------------------------
pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)

with open(SIG_LASSO_JSON, "w") as f:
    json.dump(signatures_lasso, f, indent=2)

with open(SIG_RFE_JSON, "w") as f:
    json.dump(signatures_rfe, f, indent=2)

print(f"\n\n🎉 OK - Nested CV completado. Resultados en {RESULTS_CSV}")
print(f"Firmas LASSO: {SIG_LASSO_JSON} | Firmas SVM-RFE: {SIG_RFE_JSON}")
