#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 5: STABILITY ANALYSIS
- Lee firmas por outer split (signatures_LASSO.json, signatures_SVMRFE.json).
- Calcula estabilidad:
    * Distribución Jaccard (todas las parejas de firmas por método)
    * Índice de Kuncheva (corrige por tamaño de firma, requiere k constante)
- Exporta CSV con métricas y tablas de frecuencias.
"""

import json
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("results")
SIG_LASSO = RESULTS_DIR / "signatures_LASSO.json"
SIG_RFE   = RESULTS_DIR / "signatures_SVMRFE.json"

OUT_SUMMARY = RESULTS_DIR / "stability_summary.csv"

# Número total de genes p (para Kuncheva)
X_TRAIN = Path("01_Data/processed/X_train.csv")

def jaccard(a, b):
    a, b = set(a), set(b)
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def kuncheva_index(signatures, p):
    """
    Requiere k constante (todas las firmas con el mismo tamaño).
    K = (mean(|Si ∩ Sj|) - k^2/p) / (k - k^2/p)
    """
    S = [set(s) for s in signatures]
    if len(S) < 2:
        return np.nan
    k_set = {len(s) for s in S}
    if len(k_set) != 1:
        return np.nan  # no comparable si difiere tamaño
    k = k_set.pop()
    pairs = list(itertools.combinations(S, 2))
    overlaps = [len(a & b) for a, b in pairs]
    ov_bar = np.mean(overlaps)
    expected = (k**2) / p
    denom = (k - expected)
    return (ov_bar - expected) / denom if denom != 0 else np.nan

def freq(signatures):
    flat = [g for sig in signatures for g in sig]
    c = Counter(flat)
    n = len(signatures) if signatures else 1
    df = pd.DataFrame([{"gene": g, "freq": v, "pct": v/n} for g, v in c.items()])\
         .sort_values(["freq","gene"], ascending=[False, True])
    return df

def method_stats(name, signatures, p):
    # Jaccard distribution
    pairs = list(itertools.combinations(range(len(signatures)), 2))
    j_vals = [jaccard(signatures[i], signatures[j]) for i, j in pairs] if pairs else []
    kun = kuncheva_index(signatures, p)
    return {
        "method": name,
        "n_signatures": len(signatures),
        "median_jaccard": np.median(j_vals) if j_vals else np.nan,
        "iqr_jaccard": (np.percentile(j_vals, 75) - np.percentile(j_vals, 25)) if len(j_vals) > 0 else np.nan,
        "kuncheva": kun
    }, pd.Series(j_vals)

def main():
    lasso = json.load(open(SIG_LASSO)) if SIG_LASSO.exists() else []
    rfe   = json.load(open(SIG_RFE)) if SIG_RFE.exists() else []
    p = pd.read_csv(X_TRAIN, nrows=1, index_col=0).shape[1]

    rows = []
    jaccard_dump = {}

    for name, S in [("LASSO", lasso), ("SVM-RFE", rfe)]:
        mrow, jv = method_stats(name, S, p)
        rows.append(mrow)
        jaccard_dump[name] = jv.tolist()

        # Frecuencias
        freq(S).to_csv(RESULTS_DIR / f"frequency_{name.replace('-','')}.csv", index=False)

    pd.DataFrame(rows).to_csv(OUT_SUMMARY, index=False)
    with open(RESULTS_DIR / "jaccard_values.json", "w") as f:
        json.dump(jaccard_dump, f, indent=2)

    print("OK - Estabilidad exportada en", OUT_SUMMARY)

if __name__ == "__main__":
    main()
