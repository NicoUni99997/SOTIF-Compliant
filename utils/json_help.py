import os
import shutil
import json
from pathlib import Path
from typing import Optional


def copy_json_logs(scenario, log_root):
    log_dir = os.path.join(log_root, f"gen_{scenario.gid}_scenario_{scenario.cid}")
    os.makedirs(log_dir, exist_ok=True)

    json_dir = os.path.join("temp_dir", "json")
    if os.path.exists(json_dir):
        for fname in os.listdir(json_dir):
            if fname.endswith(".json") or fname.endswith(".rec"):
                print(f"Found in dir: {fname}")
                shutil.copy2(os.path.join(json_dir, fname),
                             os.path.join(log_dir, fname))

        # rimuovi la cartella json dopo la copia
        try:
            if os.path.abspath(json_dir) != os.path.abspath(log_dir):
                shutil.rmtree(json_dir)
            else:
                print(f"Skip delete: {json_dir} == {log_dir}")
        except Exception as e:
            print(f"Errore rimuovendo {json_dir}: {e}")

def load_json(path: Path) -> dict:
    """
    Carica un file JSON dal path passato.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path) -> None:
    """
    Salva un dict come JSON formattato.    
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8"
    )


def build_output_path(input_path: Path, input_dir: Path, output_dir: Optional[Path]) -> Path:
    """
    Costruisce il path di output a partire dal file di input.
    Se è specificata output_dir, mantiene la stessa struttura relativa.
    Altrimenti aggiunge il suffisso '_with_metrics.json'.
    """
    if output_dir:
        rel_path = input_path.relative_to(input_dir)
        return output_dir / rel_path
    return input_path.with_name(input_path.stem + "_with_metrics.json")