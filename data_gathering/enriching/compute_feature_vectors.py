"""
compute_feature_vectors_multi_fast.py
RQ3: Estrae feature vectors da log arricchiti in datasets/<tool_name>/...

- Cerca *_log_basic.json ricorsivamente sotto ogni sottocartella di datasets/
- Per ogni tool:
  - scrive subito le righe run-level: <tool>_feature_vectors_runs.csv
  - calcola incrementalmente scenario-level (mean per continue, sum per count, min per time_to_first_*)
    e salva: <tool>_feature_vectors_scenarios.csv
- Logging di avanzamento per non impazzire.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict

from feature_extraction import EnrichedLogFeatureExtractorFast, FeatureVector


# Colonne: definisci qui l'ordine per CSV run-level
RUN_FIELDS = [
    "scenario_id", "tool", "generation_id", "map_name", "run_index",
    "mean_speed", "max_speed", "mean_long_acc", "p95_long_acc", "max_long_acc",
    "min_ttc", "mdbv", "tet_total", "tet_max",
    "collision_count", "red_light_count", "stop_sign_count", "speeding_count",
    "lane_invasion_count", "off_road_count",
    "time_to_first_hazard", "time_to_first_collision", "time_to_first_lane_invasion", "time_to_first_off_road",
    "completion_rate", "actual_distance_traveled", "max_progress_reached",
    "odd_global", "tc_count",
]

# Tipi di aggregazione per scenario-level
COUNT_FIELDS = {
    "collision_count", "red_light_count", "stop_sign_count", "speeding_count",
    "lane_invasion_count", "off_road_count", "tc_count",
}
# Per i time-to-first ha più senso prendere il MIN (più precoce = più critico)
MIN_FIELDS = {
    "time_to_first_hazard", "time_to_first_collision", "time_to_first_lane_invasion", "time_to_first_off_road"
}
# Tutto il resto numerico -> mean
MEAN_FIELDS = [f for f in RUN_FIELDS if f not in {"scenario_id", "tool", "generation_id", "map_name", "run_index"} | COUNT_FIELDS | MIN_FIELDS]


def fv_to_row(fv: FeatureVector) -> Dict[str, Any]:
    d = fv.__dict__.copy()
    # assicurati che tutti i campi esistano
    return {k: d.get(k, None) for k in RUN_FIELDS}


class ScenarioAgg:
    """Aggregatore scenario-level incremental."""
    __slots__ = ("n", "sum", "min", "max")

    def __init__(self):
        self.n = 0
        self.sum: Dict[str, float] = defaultdict(float)
        self.min: Dict[str, float] = {}
        self.max: Dict[str, float] = {}

    def update(self, row: Dict[str, Any]):
        self.n += 1

        # mean fields: somma
        for f in MEAN_FIELDS:
            v = row.get(f, None)
            if v is None or v == "":
                continue
            try:
                self.sum[f] += float(v)
            except Exception:
                pass

        # count fields: somma int
        for f in COUNT_FIELDS:
            v = row.get(f, 0)
            try:
                self.sum[f] += float(v)
            except Exception:
                pass

        # min fields: min (ignorando None)
        for f in MIN_FIELDS:
            v = row.get(f, None)
            if v is None or v == "":
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            if f not in self.min or vv < self.min[f]:
                self.min[f] = vv

    def finalize(self, id_part: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(id_part)
        # mean
        for f in MEAN_FIELDS:
            out[f] = (self.sum.get(f, 0.0) / self.n) if self.n > 0 else None
        # counts
        for f in COUNT_FIELDS:
            out[f] = int(round(self.sum.get(f, 0.0)))
        # mins
        for f in MIN_FIELDS:
            out[f] = self.min.get(f, None)
        return out


def process_tool(tool_dir: Path, out_dir: Path, pattern: str = "*_log_basic.json"):
    tool_name = tool_dir.name
    logs = sorted(tool_dir.rglob(pattern))
    if not logs:
        print(f"[WARN] {tool_name}: nessun log trovato ({pattern}) in {tool_dir}")
        return

    print("\n----------------------------------------------")
    print(f"[TOOL] {tool_name}")
    print(f"[INFO] Directory: {tool_dir.resolve()}")
    print(f"[INFO] Log trovati: {len(logs)}")
    print("----------------------------------------------", flush=True)


    extractor = EnrichedLogFeatureExtractorFast()

    out_runs = out_dir / f"{tool_name}_feature_vectors_runs.csv"
    out_scen = out_dir / f"{tool_name}_feature_vectors_scenarios.csv"

    # Aggregazione per scenario_id + tool + generation_id + map_name
    aggs: Dict[tuple, ScenarioAgg] = {}

    # Scrittura streaming run-level
    with out_runs.open("w", newline="") as f:
        print(f"[INFO] {tool_name}: scrittura CSV run-level -> {out_runs.name}", flush=True)
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()

        skipped = 0
        for i, p in enumerate(logs, start=1):
            try:
                obj = extractor.parse_log(p.read_text(errors="ignore"))
                fv = extractor.extract(obj)
                row = fv_to_row(fv)
                w.writerow(row)

                key = (row["scenario_id"], row["tool"], row["generation_id"], row["map_name"])
                if key not in aggs:
                    aggs[key] = ScenarioAgg()
                aggs[key].update(row)

            except Exception as e:
                skipped += 1

            if i % 25 == 0 or i == len(logs):
                print(f"[INFO] {tool_name}: processati {i}/{len(logs)} (skipped={skipped})")

    # Scrittura scenario-level (pochi record, quindi ok tutto in RAM)
    scen_fields = ["scenario_id", "tool", "generation_id", "map_name"] + MEAN_FIELDS + sorted(COUNT_FIELDS) + sorted(MIN_FIELDS)
    with out_scen.open("w", newline="") as f:
        print(f"[INFO] {tool_name}: scrittura CSV scenario-level -> {out_scen.name}", flush=True)
        w = csv.DictWriter(f, fieldnames=scen_fields)
        w.writeheader()
        for (scenario_id, tool, generation_id, map_name), agg in aggs.items():
            row = agg.finalize({
                "scenario_id": scenario_id,
                "tool": tool,
                "generation_id": generation_id,
                "map_name": map_name,
            })
            w.writerow(row)

    print("==============================================")
    print(f"[DONE] {tool_name}")
    print(f"[DONE] Run-level CSV: {out_runs.resolve()}")
    print(f"[DONE] Scenario-level CSV: {out_scen.resolve()}")
    print("==============================================\n", flush=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_dir", required=True, help="Cartella datasets/ contenente sottocartelle per tool")
    ap.add_argument("--out_dir", default=None, help="Dove salvare i CSV (default: datasets_dir)")
    ap.add_argument("--pattern", default="*_log_basic.json", help="Pattern dei log arricchiti")
    args = ap.parse_args()

    datasets_dir = Path(args.datasets_dir)
    if not datasets_dir.exists():
        raise FileNotFoundError(f"datasets_dir non trovato: {datasets_dir}")
    
    out_dir = Path(args.out_dir) if args.out_dir else datasets_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_dirs = [p for p in datasets_dir.iterdir() if p.is_dir()]
    if not tool_dirs:
        raise RuntimeError(f"Nessuna sottocartella tool trovata in {datasets_dir}")

    print("==============================================")
    print("[START] Estrazione feature vectors (RQ3)")
    print(f"[INFO] Datasets dir: {datasets_dir.resolve()}")
    print(f"[INFO] Output dir: {out_dir.resolve()}")
    print(f"[INFO] Tool trovati: {[d.name for d in tool_dirs]}")
    print("==============================================", flush=True)
    
    for td in tool_dirs:
        process_tool(td, out_dir, pattern=args.pattern)

    print("##############################################")
    print("[DONE] Estrazione feature vectors COMPLETATA")
    print("##############################################", flush=True)



if __name__ == "__main__":
    main()
