import json
import csv
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List

# ---------------------------------------------------------------------
# Hazard considerati (coerenti con i log base)
# ---------------------------------------------------------------------
HAZARDS = [
    "collision_vehicle",
    "collision_pedestrian",
    "collision_static",
    "red_light",
    "stop_sign",
    "off_road",
    "lane_invasion",
]

# ---------------------------------------------------------------------
# Severità (CARLA Leaderboard + estensione minima)
# ---------------------------------------------------------------------
SEVERITY_WEIGHTS = {
    # CARLA Leaderboard
    "collision_pedestrian": 0.50,
    "collision_vehicle": 0.60,
    "collision_static": 0.65,
    "red_light": 0.70,
    "stop_sign": 0.80,

    # Estensione minima motivata
    "off_road": 0.90,
    "lane_invasion": 0.85,
}

# ---------------------------------------------------------------------
# Parsing filename
# Expect: <scenario_id>_run_<NN>_log_basic.json
# ---------------------------------------------------------------------
def parse_filename(path: Path):
    regex = r"(.+)_run_(\d+)_log_basic\.json"
    m = re.match(regex, path.name)
    if not m:
        return None, None
    scenario_id = m.group(1)
    run_id = int(m.group(2))
    return scenario_id, run_id


# ---------------------------------------------------------------------
# Estrae i contatori degli hazard da un log base
# ---------------------------------------------------------------------
def extract_hazard_counts(data: Dict[str, Any]) -> Dict[str, int]:
    results = data.get("results", {})
    counts = results.get("event_counts", {})
    return {h: int(counts.get(h, 0)) for h in HAZARDS}


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute SOTIF Hazard (Leaderboard)")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path alla cartella dataset da analizzare"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir non trovata: {dataset_dir}")

    # Root progetto (empirical-scengen-comparison)
    base_dir = dataset_dir.parents[1]

    dataset_name = dataset_dir.name

    # Input logs (o file intermedi) dentro la cartella dataset
    logs_dir = dataset_dir

    # Output CSV dinamico (uno per dataset)
    out_csv = base_dir / "datasets" / f"{dataset_name}_sotif_hazard_leaderboard.csv"

    print(f"[INFO] Dataset        : {dataset_name}")
    print(f"[INFO] Input logs     : {logs_dir}")
    print(f"[INFO] Output CSV     : {out_csv}")

    files = sorted(logs_dir.glob("*_run_*_log_basic.json"))
    if not files:
        raise SystemExit(f"Nessun *_log_basic.json trovato in {logs_dir}")

    # Raggruppa per scenario
    grouped: Dict[str, List[Path]] = {}
    for f in files:
        sid, rid = parse_filename(f)
        if sid is None:
            print(f"[WARN] Filename non riconosciuto: {f.name}")
            continue
        grouped.setdefault(sid, []).append(f)

    # Header CSV
    header = [
        "scenario_id",
        "num_runs",
    ]

    for h in HAZARDS:
        header.append(f"P_{h}")
    for h in HAZARDS:
        header.append(f"S_{h}")
    for h in HAZARDS:
        header.append(f"R_{h}")

    rows = []

    # -----------------------------------------------------------------
    # Per ogni scenario
    # -----------------------------------------------------------------
    for scenario_id, runs in grouped.items():
        N = len(runs)

        # Conteggio run in cui l’hazard si verifica almeno una volta
        hazard_run_counts = {h: 0 for h in HAZARDS}

        for p in runs:
            with p.open() as f:
                data = json.load(f)

            counts = extract_hazard_counts(data)

            for h in HAZARDS:
                if counts[h] > 0:
                    hazard_run_counts[h] += 1

        # Probabilità empirica
        P = {h: hazard_run_counts[h] / N for h in HAZARDS}

        # Severità fissa (Leaderboard)
        S = {h: SEVERITY_WEIGHTS[h] for h in HAZARDS}

        # Rischio residuo
        R = {h: P[h] * S[h] for h in HAZARDS}

        row = [scenario_id, N]
        for h in HAZARDS:
            row.append(round(P[h], 4))
        for h in HAZARDS:
            row.append(S[h])
        for h in HAZARDS:
            row.append(round(R[h], 4))

        rows.append(row)

    # Scrittura CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fw:
        writer = csv.writer(fw)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    print(f"[OK] Hazard SOTIF (Leaderboard) calcolato per {len(rows)} scenari.")
    print(f"[OK] CSV scritto in: {out_csv}")


if __name__ == "__main__":
    main()
