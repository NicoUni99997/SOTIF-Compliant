import math
import os
import time
import subprocess
from typing import Dict, Optional, Tuple

import carla
import psutil
import logging


# Configurazione logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Path CARLA (meglio sovrascriverlo via env)
CARLA_PATH = os.environ.get("CARLA_PATH")

def is_carla_running(host: str = "localhost", port: int = 2000, timeout: float = 5.0) -> bool:
    """
    Verifica se un server CARLA è in esecuzione sulla porta indicata.
    """
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        _ = client.get_world()  # ping
        return True
    except Exception:
        return False

def kill_carla():
    """Termina eventuali processi CarlaUE4 attivi."""
    killed = 0
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            name = proc.info.get("name", "")
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "CarlaUE4" in name or "CarlaUE4.sh" in cmdline:
                logger.info(f"Killing old CarlaUE4 (pid={proc.info['pid']})")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed


def restart_carla(port: int = 2000, sleep_time: float = 20.0):
    """Riavvia CARLA su una porta specifica."""
    killed = kill_carla()
    if killed:
        logger.info(f"{killed} processi CARLA terminati")
        time.sleep(3)  # cleanup socket

    logger.info(f"Restarting CARLA on port {port}...")
    if CARLA_PATH is None:
        raise ValueError("CARLA_PATH non definito. Impostalo tramite variabile d'ambiente o hardcoded.")

    subprocess.Popen(
        [CARLA_PATH, f"-carla-port={port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    # attesa iniziale per boot UE4
    time.sleep(sleep_time)


def connect_to_carla(
    town: str,
    host: str = "localhost",
    port: int = 2000,
    max_retries: int = 10,
    retry_delay: float = 5.0,
    auto_start: bool = False,
    settings: Optional[carla.WorldSettings] = None
) -> Tuple[Optional[carla.World], Optional[carla.Client]]:
    """
    Connette a CARLA e restituisce un world configurato.
    Se auto_start=True, prova ad avviare CARLA se non è già in esecuzione.
    """
    attempt = 0

    while attempt < max_retries:
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)

            logger.info(f"[CARLA] load_world('{town}')")
            world: carla.World = client.load_world(town)

            # Impostazioni world
            if not settings:
                settings = world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1 / 20
                world.apply_settings(settings)
            else:
                world.apply_settings(settings)

            logger.info(f"[CARLA] Connesso e world pronto su {host}:{port}")
            return world, client

        except Exception as e:
            logger.warning(
                f"[CARLA][Tentativo {attempt+1}/{max_retries}] Connessione fallita: {e}"
            )

            if auto_start and attempt == 0:
                logger.info("[CARLA] Avvio automatico di CARLA...")
                restart_carla(port=port)

            attempt += 1
            if attempt < max_retries:
                logger.info(f"[CARLA] Riprovo tra {retry_delay:.1f}s...")
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Impossibile connettersi a CARLA dopo {max_retries} tentativi."
                )
                return None, None
            
    return None, None

def clean_up(
    world: carla.World,
    vehicle_list: Optional[list] = None,
    sensors: Optional[dict] = None,
) -> dict:
    """
    Cleanup centralizzato di sensori, veicoli e pedoni.
    Parametri:
      - world: oggetto carla.World
      - vehicle_list: lista di tuple o actor (es. [(idx, actor, info), ...] o [actor, ...])
      - sensors: dizionario opzionale con chiavi nominate
    Ritorna:
      report dict con conteggi distrutti e eventuali errori.
    """

    report = {
        "sensors_destroyed": 0,
        "vehicles_destroyed": 0,
        "walkers_destroyed": 0,
        "errors": [],
    }

    logger.info("Iniziando cleanup...")

    # distruggi sensori passati in sensors (se presenti)
    if sensors:
        for name, sensor in sensors.items():
            if sensor is None:
                continue
            try:
                # some objects may not expose is_alive; be permissivi
                is_alive = getattr(sensor, "is_alive", True)
                if is_alive:
                    sensor.destroy()
                    logger.info(f"{name} distrutto")
                    report["sensors_destroyed"] += 1
                else:
                    logger.debug(f"{name} non vivo. Skip destroy.")
            except Exception as e:
                msg = f"Errore cleanup {name}: {e}"
                logger.error(msg)
                report["errors"].append(msg)

    # distruggi veicoli nella lista (supporta tuple come (idx, actor, info))
    if vehicle_list:
        try:
            for item in vehicle_list:
                actor = None
                # support tuple/list where actor è il secondo elemento
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    actor = item[1]
                else:
                    actor = item

                if actor is None or not isinstance(actor, carla.Actor):
                    continue

                try:
                    is_alive = getattr(actor, "is_alive", True)
                    if is_alive:
                        actor.destroy()
                        logger.info(f"Destroyed actor id={getattr(actor, 'id', 'n/a')}")
                        report["vehicles_destroyed"] += 1
                    else:
                        logger.debug(f"Actor id={getattr(actor, 'id', 'n/a')} non vivo. Skip.")
                except Exception as e:
                    msg = f"Errore distruggendo actor id={getattr(actor, 'id', 'n/a')}: {e}"
                    logger.error(msg)
                    report["errors"].append(msg)
        except Exception as e:
            msg = f"Errore iterazione vehicle_list: {e}"
            logger.error(msg)
            report["errors"].append(msg)

    try:
        walker_controllers = world.get_actors().filter("controller.ai.walker")
        for ctrl in walker_controllers:
            if ctrl.is_alive:
                ctrl.stop()
                ctrl.destroy()
                print(f"Destroyed walker controller {ctrl.id}")
    except Exception as e:
        print(f"Errore cleanup walker controllers: {e}")

    # distruggi walkers trovati nel world
    try:
        walker_actors = world.get_actors().filter("walker.pedestrian.*")
        logger.info(f"walker_list length: {len(walker_actors)}")
        for walker in walker_actors:
            try:
                is_alive = getattr(walker, "is_alive", True)
                if is_alive:
                    walker.destroy()
                    report["walkers_destroyed"] += 1
                    logger.info(f"Destroyed walker {walker.id}")
            except Exception as e:
                msg = f"Errore cleanup walker id={getattr(walker, 'id', 'n/a')}: {e}"
                logger.error(msg)
                report["errors"].append(msg)
    except Exception as e:
        msg = f"Errore recupero walkers dal world: {e}"
        logger.error(msg)
        report["errors"].append(msg)

    return report

def analyze_collision(event: carla.CollisionEvent) -> Dict:
    ego = event.actor
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_vel = ego.get_velocity() #m/s
    ego_speed_mps = math.sqrt(ego_vel.x ** 2 + ego_vel.y ** 2 + ego_vel.z ** 2)

    other = event.other_actor
    other_tf = other.get_transform()
    other_loc = other_tf.location
    other_vel = other.get_velocity() #m/s
    other_speed_mps = math.sqrt(other_vel.x ** 2 + other_vel.y ** 2 + other_vel.z ** 2)

    frame = event.frame
    timestamp = event.timestamp

    # 1. Estrai impulso
    impulse = event.normal_impulse
    impulse_mag = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)

    # 2. Classificazione da impulso (se disponibile)
    if impulse_mag > 1e-3:
        lon_i, lat_i = project_to_ego_frame_2d(ego_tf, impulse.x, impulse.y)
        side_by_impulse, angle_by_impulse = classify_impact_direction(lon_i, lat_i)
    else:
        side_by_impulse, angle_by_impulse = None, None

    # 3. Classificazione da posizione relativa (fallback)
    rel_x = other_loc.x - ego_loc.x
    rel_y = other_loc.y - ego_loc.y
    lon_p, lat_p = project_to_ego_frame_2d(ego_tf, rel_x, rel_y)
    side_by_position, angle_by_position = classify_impact_direction(lon_p, lat_p)

    # 4. Scelta finale (priorità: impulso > posizione)
    if side_by_impulse:
        impact_side = side_by_impulse
        impact_angle = angle_by_impulse
        method_used = "impulse"
    else:
        impact_side = side_by_position
        impact_angle = angle_by_position
        method_used = "relative_position"
    
    collision_data = {
        "frame": frame, 
        "timestamp": timestamp,
        "actors_involved": {
            "ego": {
                "type_id": ego.type_id,
                "id": ego.id, 
                "location": [ego_loc.x, ego_loc.y, ego_loc.z],
                "speed_mps": ego_speed_mps
            },
            "other_actor": {
                "type_id": other.type_id,
                "id": other.id,
                "location": [other_loc.x, other_loc.y, other_loc.z],
                "speed_mps": other_speed_mps
            }
        },
        "impact": {
            "impulse": {
                "vector": [impulse.x, impulse.y, impulse.z],
                "magnitude": impulse_mag
            },
            "impact_side": impact_side,
            "impact_angle_deg": impact_angle,
            "classification_method": method_used
        }
    }

    return collision_data
    
    
def project_to_ego_frame_2d(ego_transform: carla.Transform, vec_x: float, vec_y: float) -> Tuple[float, float]:
    """
    Proietta un vettore 2D (vec_x, vec_y), espresso nel sistema di riferimento globale,
    nel sistema locale del veicolo ego (forward/lateral).

    Returns:
        Tuple[float, float]: (componente longitudinale, componente laterale)
    """
    yaw_rad = math.radians(ego_transform.rotation.yaw)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)

    # Proiezione su assi ego (matrice di rotazione inversa)
    longitudinal = cos_yaw * vec_x + sin_yaw * vec_y
    lateral = -sin_yaw * vec_x + cos_yaw * vec_y

    return longitudinal, lateral


def classify_impact_direction(longitudinal: float, lateral: float) -> Tuple[str, float]:
    """
    Classifica la direzione di un impatto rispetto al veicolo ego in:
    - 'Fronte', 'Retro', 'Lato sinistro', 'Lato destro'

    Returns:
        Tuple[str, float]: (etichetta lato, angolo in gradi)
    """
    angle_deg = math.degrees(math.atan2(lateral, longitudinal))  # [-180°, +180°]
    abs_angle = abs(angle_deg)

    if abs_angle <= 45:
        label = "Fronte"
    elif abs_angle >= 135:
        label = "Retro"
    else:
        label = "Lato destro" if angle_deg > 0 else "Lato sinistro"

    return label, angle_deg


def get_speed_mps(ego_vehicle: carla.Vehicle):
    velocity = ego_vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

def get_speed_kmh(ego_vehicle: carla.Vehicle):
    velocity = ego_vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

FOLLOW = 6

def follow_ego(spec: carla.Actor, ego: carla.Vehicle):
    location = ego.get_location()
    rotation = ego.get_transform().rotation
    fwd_vec = rotation.get_forward_vector()

    location.x -= FOLLOW * fwd_vec.x
    location.y -= FOLLOW * fwd_vec.y
    location.z += 3  
    rotation.pitch -= 5  
    spec.set_transform(
        carla.Transform(location, rotation)
    )