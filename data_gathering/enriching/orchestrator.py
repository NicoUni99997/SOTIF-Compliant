import argparse
import json
import sys
import traceback
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from critical import calculate_scenario_metrics
from dynamic import DynamicsAnalyzer
from functional import FunctionalAnalyzer
from log_normalization import normalize_events, ensure_event_counts_schema

sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.json_help import load_json, save_json, build_output_path

def process_single_file(
    input_path: Path,
    input_dir: Path,
    output_dir: Optional[Path],
    completion_tolerance: float = 10.0,
    stability_threshold: float = 5.0
) -> Optional[str]:
    try:
        # Caricamento JSON
        log_data = load_json(input_path)

        # Normalizzazione schema/eventi (frame/timestamp/naming)
        # Serve a rendere coerenti i dati prima dell'arricchimento.
        normalize_events(log_data)
        ensure_event_counts_schema(log_data)

        frames = log_data.get("frames", [])
        delta_time = float(log_data.get("delta_time", 0.05))

        # Output path
        output_path = build_output_path(input_path, input_dir, output_dir)

        # Calcolo metriche
        criticality = calculate_scenario_metrics(frames, delta_time=delta_time)

        performance = FunctionalAnalyzer(
            output_dir=output_path.parent,
            completion_tolerance=completion_tolerance,
            stability_threshold=stability_threshold
        ).analyze_to_dict(log_data)

        dynamics = DynamicsAnalyzer(frames, delta_time=delta_time).analyze()

        # Integrazione
        results = log_data.setdefault("results", {})
        results["critical_metrics"] = criticality
        results["functional_metrics"] = performance
        results["dynamics_metrics"] = dynamics

        # Salvataggio
        save_json(log_data, output_path)

        return str(output_path)
    except json.JSONDecodeError as e:
        print(f"[FAIL-JSON] {input_path.name}: {e}")
        return None
    except Exception as e:
        print(f"[FAIL] {input_path.name}: {e}")
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Orchestrator: calcola tutte le metriche in parallelo")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory contenente file *_log_basic.json")
    parser.add_argument("--output_dir", type=Path, required=False, help="Directory in cui salvare i JSON con metriche")
    parser.add_argument("--completion_tolerance", type=float, default=10.0, help="Tolleranza completamento (metri)")
    parser.add_argument("--stability_threshold", type=float, default=5.0, help="Soglia deviazione stabilità (metri)")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Numero di processi paralleli")
    args = parser.parse_args()

    all_files = list(args.input_dir.glob("**/*_log_basic.json"))
    print(f"Trovati {len(all_files)} file da processare.")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    process_fn = partial(
        process_single_file,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        completion_tolerance=args.completion_tolerance,
        stability_threshold=args.stability_threshold
    )

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_fn, all_files), total=len(all_files), desc="Elaborazione log"))

    valid = [r for r in results if r]
    print(f"\nFile processati con successo: {len(valid)} / {len(all_files)}")

if __name__ == "__main__":
    main()
