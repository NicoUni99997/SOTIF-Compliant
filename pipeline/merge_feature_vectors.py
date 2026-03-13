"""
merge_feature_vectors.py
RQ3: Merge automatico dei feature vectors scenario-level.

Cerca ricorsivamente in datasets_dir tutti i file:
  *_feature_vectors_scenarios.csv

Li concatena in un unico CSV:
  rq3_merged_feature_vectors_scenarios.csv

Uso:
python merge_feature_vectors.py --datasets_dir datasets
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def infer_tool_name(path: Path) -> str:
    # prova a ricavarlo dal nome file: <Tool>_feature_vectors_scenarios.csv
    name = path.name
    if name.endswith("_feature_vectors_scenarios.csv"):
        return name.replace("_feature_vectors_scenarios.csv", "")
    # fallback: nome cartella padre
    return path.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", required=True, help="Cartella datasets/ (ricorsiva)")
    ap.add_argument("--pattern", default="*_feature_vectors_scenarios.csv", help="Pattern file da mergiare")
    ap.add_argument("--out_name", default="rq3_merged_feature_vectors_scenarios.csv", help="Nome output merged")
    args = ap.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"datasets_dir non trovato: {datasets_dir}")

    files = sorted(datasets_dir.rglob(args.pattern))
    if not files:
        raise RuntimeError(f"Nessun file trovato con pattern {args.pattern} dentro {datasets_dir}")

    print(f"[INFO] Trovati {len(files)} CSV scenario-level da unire:")
    for p in files:
        print(f" - {p}")
    print("", flush=True)

    dfs = []
    for p in files:
        tool = infer_tool_name(p)
        df = pd.read_csv(p)
        if "tool" not in df.columns or df["tool"].isna().all():
            df["tool"] = tool
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    out_path = datasets_dir / args.out_name
    merged.to_csv(out_path, index=False)

    print(f"[DONE] Merged scritto: {out_path.resolve()}")
    print(f"[INFO] Righe: {len(merged)} | Colonne: {len(merged.columns)}")
    print("[INFO] Conteggio per tool:")
    print(merged["tool"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
