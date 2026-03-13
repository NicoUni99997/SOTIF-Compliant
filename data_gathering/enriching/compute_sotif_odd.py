import json
import csv
import argparse
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------
#  ODD: tabelle di mapping
# ---------------------------

WEATHER_FACTOR = {
    "ClearNoon": 1.0,
    "CloudyNoon": 0.9,
    "WetNoon": 0.8,
    "SoftRainNoon": 0.7,
    "HardRainNoon": 0.5,
}

TIME_OF_DAY_FACTOR = {
    "noon": 1.0,
    "morning": 0.9,
    "afternoon": 0.9,
    "dawn": 0.7,
    "dusk": 0.7,
    "night": 0.4,
}

ROAD_CONDITION_FACTOR = {
    "dry": 1.0,
    "wet": 0.8,
    "snow": 0.4,
    "ice": 0.3,
}

# traffico / pedoni (densità dichiarata nello scenario)
DENSITY_FACTOR = {
    "low": 1.0,
    "medium": 0.8,
    "high": 0.5,
}


def compute_env_subscore(env: Dict[str, Any], traffic: Dict[str, Any]) -> float:
    weather = env.get("weather_preset", "ClearNoon")
    tod = env.get("time_of_day", "noon")
    road = env.get("road_condition", "dry")

    wf = WEATHER_FACTOR.get(weather, 0.6)
    tf = TIME_OF_DAY_FACTOR.get(tod, 0.7)
    rf = ROAD_CONDITION_FACTOR.get(road, 0.8)

    # opzionale: visibilità / fog
    visibility = env.get("visibility", "medium")
    if isinstance(visibility, (int, float)):
        # es: se salvi una metrica numerica di visibilità 0-1
        vis_f = max(0.0, min(1.0, float(visibility)))
    else:
        vis_map = {"good": 1.0, "medium": 0.8, "low": 0.5}
        vis_f = vis_map.get(visibility, 0.8)

    return (wf + tf + rf + vis_f) / 4.0


def compute_infra_subscore(scenario_data: Dict[str, Any]) -> float:
    """
    Approccio semplice:
    - se mission-waypoints sono molti e attraversano incroci / rotonde, ODD è più 'tirato'
    - se la mission è corta e principalmente in rettilineo, ODD più favorevole.
    Non abbiamo i flag di junction direttamente, quindi usiamo lunghezza missione come proxy.
    """
    mission = scenario_data.get("mission", {})
    waypoints = mission.get("waypoints", [])

    num_wp = len(waypoints)

    if num_wp == 0:
        # missione non definita -> consideriamo neutro
        return 0.8
    elif num_wp < 20:
        return 1.0     # percorso semplice / corto
    elif num_wp < 50:
        return 0.8     # complessità media
    else:
        return 0.6     # percorso lungo/complesso (incroci, ecc.)


def compute_traffic_subscore(traffic: Dict[str, Any], scenario_data: Dict[str, Any]) -> float:
    traffic_density = traffic.get("traffic_density", "medium")
    ped_density = traffic.get("pedestrian_density", "medium")

    traf_f = DENSITY_FACTOR.get(traffic_density, 0.8)
    ped_f = DENSITY_FACTOR.get(ped_density, 0.8)

    # opzionale: aggiusta in base al numero reale di agenti nello scenario
    npc_vehicles = scenario_data.get("npc_vehicles", [])
    peds = scenario_data.get("pedestrians", [])

    num_npc = len(npc_vehicles)
    num_peds = len(peds)

    # se davvero pochi agenti, boost leggermente l'ODD
    if num_npc + num_peds <= 2:
        extra = 0.1
    elif num_npc + num_peds >= 10:
        extra = -0.1
    else:
        extra = 0.0

    score = (traf_f + ped_f) / 2.0 + extra
    return max(0.0, min(1.0, score))


def compute_operational_subscore(scenario_data: Dict[str, Any]) -> float:
    """
    Operational ODD:
    - velocità prevista
    - durata missione
    """
    ego = scenario_data.get("ego", {})
    speed_limit = ego.get("speed_limit", 13.9)  # m/s, ~50 km/h

    # assumiamo che 'simulation.timeout_seconds' esista nello scenario
    sim = scenario_data.get("simulation", {})
    timeout_s = sim.get("timeout_seconds", 30)

    # 1) velocità (entro 50 km/h = 1.0; oltre 80 km/h = 0.5)
    speed_kmh = float(speed_limit) * 3.6
    if speed_kmh <= 50:
        sf = 1.0
    elif speed_kmh <= 80:
        sf = 0.8
    else:
        sf = 0.5

    # 2) durata missione (più lunga -> ODD più sfidante)
    if timeout_s <= 20:
        tf = 1.0
    elif timeout_s <= 60:
        tf = 0.8
    else:
        tf = 0.6

    return (sf + tf) / 2.0


# ---------------------------
#  Triggering Conditions
# ---------------------------

def compute_triggering_conditions(
    scenario_data: Dict[str, Any],
    metrics: Dict[str, Any]
) -> List[str]:

    tcs = []
    env = scenario_data.get("environment", {})
    traffic = scenario_data.get("traffic", {})
    npc_vehicles = scenario_data.get("npc_vehicles", [])
    peds = scenario_data.get("pedestrians", [])
    mission = scenario_data.get("mission", {})

    weather = env.get("weather_preset", "ClearNoon")
    tod = env.get("time_of_day", "noon")
    road_cond = env.get("road_condition", "dry")

    # METRICHE STRUTTURALI
    min_dist = metrics.get("min_distance", None)
    min_ttc = metrics.get("min_ttc", None)
    max_ego_speed = metrics.get("max_ego_speed", None)
    avg_ego_speed = metrics.get("avg_ego_speed", None)

    # === ENVIRONMENTAL TC ===

    # TC1 - Adverse Environment (trigger SOTIF)
    if weather in ("HardRainNoon", "SoftRainNoon"):
        tcs.append("adverse_weather")
    if tod in ("dawn", "dusk", "night"):
        tcs.append("low_visibility_time")
    if road_cond in ("wet", "snow", "ice"):
        tcs.append("low_friction_surface")

    # se due condizioni ambientali critiche coesistono → super-TC
    if (
        (weather in ("HardRainNoon", "SoftRainNoon"))
        and tod in ("dawn", "dusk", "night")
    ):
        tcs.append("combined_environment_challenge")

    # === TRAFFIC COMPLEXITY TC ===

    # TC2 - High traffic interactions
    if traffic.get("traffic_density", "medium") == "high" or len(npc_vehicles) >= 8:
        tcs.append("high_traffic_density")

    # TC3 - Mixed actor crowd (pedoni + veicoli)
    if len(npc_vehicles) >= 3 and len(peds) >= 2:
        tcs.append("mixed_actor_interaction")

    # Aggressive NPC → sempre TC
    if any(v.get("behavior") == "aggressive" for v in npc_vehicles):
        tcs.append("aggressive_actor_present")

    # === DYNAMIC TC (derivate dalle metriche) ===

    # TC4 - Near miss (SOTIF classic)
    try:
        if min_dist is not None and float(min_dist) < 3.0:  # prima 2.0 (troppo conservativo)
            tcs.append("near_miss_distance")
    except:
        pass

    # TC5 - Critical TTC
    try:
        if min_ttc is not None and float(min_ttc) < 3.0:
            tcs.append("critical_ttc")
    except:
        pass

    # TC6 - High-speed approach
    try:
        if max_ego_speed is not None and float(max_ego_speed) > 12.0:  # > ~43 km/h
            if min_ttc is not None and float(min_ttc) < 4.0:
                tcs.append("high_speed_approach")
    except:
        pass

    # === INFRASTRUCTURE TC ===

    # TC7 - Complex route
    wp_len = len(mission.get("waypoints", []))
    if wp_len > 50:
        tcs.append("complex_route")
    if wp_len > 80:
        tcs.append("very_complex_route")

    tcs = sorted(set(tcs))
    return tcs


# ---------------------------
#  Main: elabora tutti i JSON
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SOTIF ODD metrics")
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path alla cartella dataset da analizzare"
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir non trovata: {dataset_dir}")

    # base_dir = root progetto (empirical-scengen-comparison)
    base_dir = dataset_dir.parents[1]

    dataset_name = dataset_dir.name

    # Input: log base
    metrics_dir = dataset_dir

    # Output: CSV dinamico
    out_csv = base_dir / "datasets" / f"{dataset_name}.csv"

    print(f"[INFO] Dataset       : {dataset_name}")
    print(f"[INFO] Input logs    : {metrics_dir}")
    print(f"[INFO] Output CSV    : {out_csv}")

    if not metrics_dir.exists():
        raise SystemExit(f"Directory metrics non trovata: {metrics_dir}")

    header = [
        "scenario_id",
        "odd_env",
        "odd_infra",
        "odd_traffic",
        "odd_operational",
        "odd_global",
        "triggering_conditions",
    ]

    files = sorted(metrics_dir.glob("*_log_basic.json"))
    if not files:
        raise SystemExit(f"Nessun *_log_basic.json trovato in {metrics_dir}")

    total_files = len(files)
    print(f"[STEP A] Trovati {total_files} file in {metrics_dir}")

    rows = []
    seen_scenarios = set()

    for idx, path in enumerate(files, start=1):
        fname = path.name
        print(f"[STEP A] ({idx}/{total_files}) Elaboro: {fname}")

        with path.open() as f:
            data = json.load(f)

        scenario_id = data.get("scenario_id", path.stem)
        seen_scenarios.add(scenario_id)

        # struttura generica:
        # - scenario_data con mission, environment, traffic, ego, simulation, ecc.
        scenario_data = data.get("scenario_data", data)

        env = scenario_data.get("environment", {})
        traffic = scenario_data.get("traffic", {})

        odd_env = compute_env_subscore(env, traffic)
        odd_infra = compute_infra_subscore(scenario_data)
        odd_traffic = compute_traffic_subscore(traffic, scenario_data)
        odd_oper = compute_operational_subscore(scenario_data)

        odd_global = (odd_env + odd_infra + odd_traffic + odd_oper) / 4.0

        # Metriche arricchite: nel nostro schema stanno sotto results.*_metrics
        results = data.get("results", {}) or {}
        crit = results.get("critical_metrics", {}) or {}
        dyn = results.get("dynamics_metrics", {}) or {}

        # Mappa compatibile con compute_triggering_conditions
        metrics = {
            "min_distance": crit.get("MDBV"),
            "min_ttc": crit.get("min_TTC"),
            "max_ego_speed": dyn.get("max_speed"),
            "avg_ego_speed": dyn.get("mean_speed"),
        }
        tcs = compute_triggering_conditions(scenario_data, metrics)

        # salva dentro il JSON
        sotif_block = data.get("sotif", {})
        sotif_block.update(
            {
                "odd_env": odd_env,
                "odd_infra": odd_infra,
                "odd_traffic": odd_traffic,
                "odd_operational": odd_oper,
                "odd_global": odd_global,
                "triggering_conditions": tcs,
            }
        )
        data["sotif"] = sotif_block

        with path.open("w") as f:
            json.dump(data, f, indent=2)

        rows.append(
            [
                scenario_id,
                odd_env,
                odd_infra,
                odd_traffic,
                odd_oper,
                odd_global,
                ";".join(tcs),
            ]
        )

    # CSV riassuntivo
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    print(f"[STEP A] File JSON elaborati: {total_files}")
    print(f"[STEP A] Scenari distinti trovati: {len(seen_scenarios)}")
    print(f"[STEP A] CSV scritto in: {out_csv}")


if __name__ == "__main__":
    main()