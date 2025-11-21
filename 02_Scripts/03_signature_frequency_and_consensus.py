#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script 3: FEATURE LIST EXTRACTION
- Lee firmas por outer split desde signatures_*.json (generadas por Script 2).
- Calcula frecuencias de selección y exporta:
    * frequency_<method>.csv (gene, freq, pct)
    * consensus_<method>_topK.csv (ej. top-N por frecuencia)
- (Opcional) Intersección y unión entre métodos.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("results")
SIG_LASSO = RESULTS_DIR / "signatures_LASSO.json"
SIG_RFE   = RESULTS_DIR / "signatures_SVMRFE.json"

TOP_K = 50  # exportar top-K por método

def freq_table(signatures):
    flat = [g for sig in signatures for g in sig]
    cnt = Counter(flat)
    n = len(signatures) if len(signatures) > 0 else 1
    df = pd.DataFrame([{"gene": g, "freq": c, "pct": c / n} for g, c in cnt.items()]).sort_values(
        ["freq", "gene"], ascending=[False, True])
    return df

def main():
    lasso = json.load(open(SIG_LASSO)) if SIG_LASSO.exists() else []
    rfe   = json.load(open(SIG_RFE)) if SIG_RFE.exists() else []

    df_lasso = freq_table(lasso)
    df_rfe   = freq_table(rfe)

    df_lasso.to_csv(RESULTS_DIR / "frequency_LASSO.csv", index=False)
    df_rfe.to_csv(RESULTS_DIR / "frequency_SVMRFE.csv", index=False)

    df_lasso.head(TOP_K).to_csv(RESULTS_DIR / f"consensus_LASSO_top{TOP_K}.csv", index=False)
    df_rfe.head(TOP_K).to_csv(RESULTS_DIR / f"consensus_SVMRFE_top{TOP_K}.csv", index=False)

    # Intersección & unión de top-K
    l_top = set(df_lasso.head(TOP_K)["gene"].tolist())
    r_top = set(df_rfe.head(TOP_K)["gene"].tolist())
    inter = sorted(list(l_top & r_top))
    union = sorted(list(l_top | r_top))

    pd.DataFrame({"gene": inter}).to_csv(RESULTS_DIR / f"intersection_top{TOP_K}.csv", index=False)
    pd.DataFrame({"gene": union}).to_csv(RESULTS_DIR / f"union_top{TOP_K}.csv", index=False)

    print("OK - Frecuencias y consensos exportados.")

if __name__ == "__main__":
    main()
