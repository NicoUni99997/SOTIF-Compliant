"""log_normalization.py

Utility per rendere *coerenti* i log base/arricchiti.

Problema reale (e fastidioso): nei log possono convivere frame "locali" (0..N)
e carla_frame (snapshot.frame, valori enormi), e timestamp sia relativi alla run
che assoluti (uptime simulatore).

Queste funzioni normalizzano:
 - nomi degli eventi (red_lights -> red_light)
 - frame degli eventi (se sembra un carla_frame lo mappiamo al frame locale)
 - timestamp degli eventi (se non plausibile, lo ricalcoliamo dai frames)
 - posizione del nodo events (duplica sotto results per compatibilità)

Non richiede riesecuzione delle simulazioni.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional


EVENT_ALIASES = {
    "red_lights": "red_light",
    "red_light": "red_light",
    "stop": "stop_sign",
    "stop_sign": "stop_sign",
}


def _as_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _as_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def build_frame_index(frames: List[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Ritorna (by_local_frame, by_carla_frame)."""
    by_local: Dict[int, Dict[str, Any]] = {}
    by_carla: Dict[int, Dict[str, Any]] = {}

    for idx, fr in enumerate(frames):
        lf = _as_int(fr.get("frame", idx))
        if lf is None:
            continue
        by_local[lf] = fr

        cf = _as_int(fr.get("carla_frame"))
        if cf is not None:
            by_carla[cf] = fr

    return by_local, by_carla


def normalize_event_key(k: str) -> str:
    return EVENT_ALIASES.get(k, k)


def normalize_events(log: Dict[str, Any], *, max_timestamp_factor: float = 10.0) -> Dict[str, Any]:
    """Normalizza log["events"] (in-place) e garantisce compatibilità.

    max_timestamp_factor: se un timestamp evento è > (simulated_time * factor) viene considerato "assurdo".
    """

    frames: List[Dict[str, Any]] = log.get("frames", []) or []
    by_local, by_carla = build_frame_index(frames)

    results = log.get("results", {}) or {}
    sim_t = _as_float(results.get("total_simulation_time"))
    if sim_t is None:
        # fallback: ultimo timestamp frame
        if frames:
            sim_t = _as_float(frames[-1].get("timestamp"))
        if sim_t is None:
            sim_t = 0.0

    events = log.get("events", {}) or {}
    normalized: Dict[str, Any] = {}

    for raw_k, ev_list in events.items():
        k = normalize_event_key(str(raw_k))
        if not isinstance(ev_list, list):
            # mantieni com'è
            normalized[k] = ev_list
            continue

        out_list = []
        for ev in ev_list:
            if not isinstance(ev, dict):
                out_list.append(ev)
                continue

            ev2 = dict(ev)

            # frame: se è un carla_frame (tipicamente molto grande), mappalo
            fr = _as_int(ev2.get("frame"))
            if fr is not None:
                # Heuristica: se fr non esiste nei local frame ma esiste nei carla frame, è un carla_frame
                if fr not in by_local and fr in by_carla:
                    ev2["carla_frame"] = fr
                    ev2["frame"] = _as_int(by_carla[fr].get("frame"))

            # timestamp: se non plausibile, ricalcola dal frame locale
            ts = _as_float(ev2.get("timestamp"))
            lf = _as_int(ev2.get("frame"))
            if lf is not None and lf in by_local:
                fr_ts = _as_float(by_local[lf].get("timestamp"))
            else:
                fr_ts = None

            if ts is None or (sim_t > 0 and ts > sim_t * max_timestamp_factor):
                if fr_ts is not None:
                    ev2["timestamp"] = fr_ts
            else:
                # se timestamp esiste ma frame_ts è vicino, ok. Se diverge tantissimo, preferisci frame_ts
                if fr_ts is not None and abs(ts - fr_ts) > max(5.0, sim_t * 0.25):
                    ev2["timestamp"] = fr_ts

            out_list.append(ev2)

        # concatena se esistono già (es. red_lights + red_light)
        if k in normalized and isinstance(normalized[k], list):
            normalized[k].extend(out_list)
        else:
            normalized[k] = out_list

    # scrivi back
    log["events"] = normalized

    # compat: duplica sotto results
    results = log.setdefault("results", {})
    results["events"] = normalized

    return log


def ensure_event_counts_schema(log: Dict[str, Any]) -> Dict[str, Any]:
    """Garantisce che results.event_counts abbia le chiavi principali usate a valle."""
    results = log.setdefault("results", {})
    counts = results.setdefault("event_counts", {})

    # alias: se esiste red_lights ma non red_light
    if "red_light" not in counts and "red_lights" in counts:
        counts["red_light"] = counts.get("red_lights", 0)
    if "stop_sign" not in counts and "stop" in counts:
        counts["stop_sign"] = counts.get("stop", 0)

    return log
