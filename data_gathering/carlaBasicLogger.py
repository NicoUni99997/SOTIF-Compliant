import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import carla

import sys
sys.path.append("/home/nico/CARLA/PythonAPI/carla/agents")
sys.path.append("/home/nico/Documenti/empirical-scengen-comparison")

#from PythonAPI.carla.agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.behavior_agent import BehaviorAgent

from utils.carla_help import analyze_collision
from violationMonitor import ViolationMonitor

LOGGER_REGISTRY: Dict[int, "CarlaBasicLogger"] = {}

class CarlaBasicLogger:
    """
        Logger incaricato di salvare per una simulazione di uno scenario un 
        corrispettivo file JSON avente al proprio interno tutte le informazioni 
        necessarie per effettuare un'analisi efficace dello scenario.
    """

    def __init__(self, tool: str, generation_id: str, scenario_id: str, output_dir: str, 
                world: carla.World, client: carla.Client, 
                violation_monitor: Optional[ViolationMonitor] = None, 
                delta_time: float = 0.05, record_binary: bool = False, run_index: int = 1):      
        """
        Inizializza il logger di base
        """
        #setting dei metadati
        self.tool = tool
        self.generation_id = generation_id
        self.scenario_id = scenario_id

        self.output_dir = output_dir
        #se non esiste la cartella di output, creala
        os.makedirs(output_dir, exist_ok=True)

        #setting temporale
        self.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.simulation_start_time = time.time()
        self._first_timestamp = None
        self.delta_time = delta_time
        #counter dei frame nel log
        self.frame_count = 0

        self.map_name = "unknown"

        self.world = world
        self.client = client
        self.record_binary = record_binary
        self.run_index = run_index
        self.recorder_path = None

        self.violation_monitor = violation_monitor

        self.last_collision_timestamp = -1 #timestamp dell'ultima collisione rilevata
        self.last_collision_actor_id = -1 #id dell'ultimo attore con cui è avvenuto una collisione
        self.collision_cooldown = 2 #secondi che devono trascorrere tra l'ultima collisione rilevata e un'altra per evitare ripetizioni

        self.ended = False #booleano che dice se deve fermarsi a loggare 
        #struttura dizionario chiave/valore che apparirà nel json finale
        self.scenario_data = {
            #metadati
            "tool": self.tool,
            "generation_id": self.generation_id,
            "scenario_id": self.scenario_id,
            "map_name": self.map_name,
            "run_index": self.run_index,
            "start_time": self.start_time,
            "simulation_start_time": self.simulation_start_time,
            "delta_time": self.delta_time,

            "results" : {
                "has_collision": False,
                "has_red_light_violation": False,
                "has_speeding": False,
                "has_stop_violation": False,
                "event_counts": {
                    "collision": 0,
                    "red_light": 0,
                    "speeding": 0,
                    "stop_sign": 0,
                }
            },
            
            #lista degli eventi principali
            "events": {
                #logging dell'eventuale collisione dell'ego con gli altri attori dinamici + relative informazioni
                "collision": [],
                #logging dell'eventuale attraversamento del semaforo rosso da parte dell'ego + relative informazioni
                "red_lights": [],
                #logging dell'eventuale superamento dei limiti di velocità da parte dell'ego + relative informazioni
                "speeding": [],
                #logging dell'eventuale non rispetto del segnale di stop (behavior agent non lo osserva -> se incluso bisogna estendere behavior agent)
                "stop_sign": [],
                "lane_invasion": [],
                "off_road": [],
            },

            #informazioni missione del veicolo ego (behavior agent)
            "mission": {
                "start_location": None,
                "end_location": None,
                "waypoints": []
            },

            #registro degli attori dinamici
            # id : { type, type_id, role, spawn_frame, spawn_transform, (despawn_frame opz.) }
            "actors": {},

            #lista di frames, ciascuno con ego + altri attori dinamici
            "frames": []
        }
        if self.tool != "TMFuzzer" and self.tool != "ScenarioFuzzLLM":
            self._init_from_world()

        self.scenario_data["results"]["event_counts"].setdefault("collision_vehicle", 0)
        self.scenario_data["results"]["event_counts"].setdefault("collision_pedestrian", 0)
        self.scenario_data["results"]["event_counts"].setdefault("collision_static", 0)
        self.scenario_data["results"]["event_counts"].setdefault("lane_invasion", 0)
        self.scenario_data["results"]["event_counts"].setdefault("off_road", 0)


    def _init_from_world(self):
        try:
            if self.world and self.client:
                # Map name
                carla_map_name = self.world.get_map().name
                self.map_name = carla_map_name.split("/")[-1]
                self.scenario_data["map_name"] = self.map_name

                # Delta time
                settings = self.world.get_settings()
                if settings.fixed_delta_seconds and settings.synchronous_mode:
                    self.delta_time = float(settings.fixed_delta_seconds)
                    self.scenario_data["delta_time"] = self.delta_time
                else:
                    print("[CarlaBasicLogger] Delta time non settato correttamente dal world.")

                # Binary recording
                if self.record_binary:
                    # path assoluto corretto
                    self.recorder_path = os.path.abspath(os.path.join(
                        self.output_dir, f"{self.generation_id}_{self.scenario_id}.rec"
                    ))
                    
                    self.client.start_recorder(self.recorder_path, True)
                    print(f"[CarlaBasicLogger] Binary recording started: {self.recorder_path}")
        except Exception as e:
            print(f"[CarlaBasicLogger] Errore durante inizializzazione da world: {e}")

    
    def set_mission_from_agent(self, ego_agent: BehaviorAgent, ego_sp, ego_dp):
        """
        Imposta le informazioni relative alla missione missione del veicolo ego nel logger.

        Args:
            ego_agent: istanza del BehaviorAgent del veicolo ego
            ego_sp: coordinate di spawn del veicolo ego
            ego_dp: coordinate della destinazione del veicolo ego
        """
        #raccolta della location di spawn e destinazione
        start_location = [ego_sp.location.x, ego_sp.location.y, ego_sp.location.z]
        end_location = [ego_dp.location.x, ego_dp.location.y, ego_dp.location.z]

        #raccolta della planned route del behavior agent
        planned_route: List[List[float]] = []
    
        try:
            #prendi il local planner dell'agente che decide la lista di waypoints da seguire per raggiungere la destinazione
            local_planner = ego_agent.get_local_planner()

            route = list(local_planner._waypoints_queue)
            for item in route:
                wpt_or_tf = item[0]
                if isinstance(wpt_or_tf, carla.Waypoint):
                    tf = wpt_or_tf.transform
                elif isinstance(wpt_or_tf, carla.Transform):
                    tf = wpt_or_tf
                planned_route.append([tf.location.x, tf.location.y, tf.location.z])
        except Exception:
            planned_route = []

        self.scenario_data["mission"]["start_location"] = list(start_location) if start_location else None
        self.scenario_data["mission"]["end_location"] = list(end_location) if end_location else None
        self.scenario_data["mission"]["waypoints"] = [list(wp) for wp in (planned_route or [])]


    def classify_actor_type(self, actor: carla.Actor) -> str:
        """
        Classifica il tipo di attore in base alle sue caratteristiche.

        Args:
            actor: istanza dell'attore da classificare

        Returns:
            str: tipo di attore ("vehicle", "pedestrian", "traffic_light", etc.)
        """
        if isinstance(actor, carla.Vehicle):
            return "vehicle"
        elif isinstance(actor, carla.Walker):
            return "pedestrian"
        elif isinstance(actor, carla.TrafficLight):
            return "traffic_light"
        else:
            return "unknown"

    def register_actor(self, actor: carla.Actor, actor_role: str, spawn_transform: List[float], snapshot: carla.WorldSnapshot):
        """
            Registra un attore dinamico
        """

        if str(actor.id) in self.scenario_data["actors"]:
            #attore già registrato
            return False
        
        if self.classify_actor_type(actor=actor) not in ["vehicle", "pedestrian"]:
            return False

        #creazione del dizionario delle informazioni statiche dell'attore
        actor_info = {
            "type_id": actor.type_id,
            "role": actor_role,
            "spawn_frame": self.frame_count,
            "spawn_carla_frame": snapshot.frame,
            "spawn_carla_timestamp": snapshot.timestamp.elapsed_seconds,
            "spawn_transform": spawn_transform
        }

        #aggiunta dell'attore al dizionario degli attori
        self.scenario_data["actors"][actor.id] = actor_info
        return True

    def register_ego_actor(self, ego_vehicle: carla.Vehicle, snapshot: carla.WorldSnapshot):
        """Registra l'ego nel logger con le sue informazioni statiche (bb + spawn_transform)"""

        transform = ego_vehicle.get_transform()

        self.register_actor(ego_vehicle, 
                            "ego",
                            [transform.location.x, transform.location.y, transform.location.z, transform.rotation.roll, transform.rotation.pitch, transform.rotation.yaw],
                            snapshot=snapshot)

    def update_frame(
            self, 
            world: carla.World, 
            ego_vehicle: carla.Vehicle,
            snapshot: Optional[carla.WorldSnapshot],
    ):
        """
        Registra un fotogramma nel logger con le informazioni sugli attori presenti nella scena.
        Args:
            world: istanza del mondo CARLA
            ego_vehicle: istanza del veicolo ego
            snapshot: istanza dell'istantanea del mondo CARLA
        """
        if self.ended:
            return
        ego_tf = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_angular_velocity = ego_vehicle.get_angular_velocity()

        ego_loc = [ego_tf.location.x, ego_tf.location.y, ego_tf.location.z]
        #roll:x, pitch:y, yaw:z
        ego_rot = [ego_tf.rotation.roll, ego_tf.rotation.pitch, ego_tf.rotation.yaw]
        ego_velocity_vec = [ego_velocity.x, ego_velocity.y, ego_velocity.z]
        ego_angular_velocity_vec = [ego_angular_velocity.x, ego_angular_velocity.y, ego_angular_velocity.z]

        #da quanto capito non affidabile (da calcolare a posteriori)
        #ego_acc = ego_vehicle.get_acceleration() 

        if snapshot is None:
            snapshot = world.get_snapshot()
        
        carla_frame = snapshot.frame
        current_timestamp = snapshot.timestamp.elapsed_seconds

        if self._first_timestamp is None:
            self._first_timestamp = current_timestamp

        relative_timestamp = current_timestamp - self._first_timestamp
        if self.violation_monitor is None:
            print("ViolationMonitor non settato correttamente: le violazioni non verranno gestite")
        else:
            self.violation_monitor.tick(carla_frame, relative_timestamp)

        frame_data = {
            "frame": self.frame_count,
            "carla_frame": carla_frame,
            "timestamp": relative_timestamp,
            "delta_time": self.delta_time,

            "ego_vehicle": {
                "location": list(ego_loc),
                "rotation": list(ego_rot),
                "velocity": list(ego_velocity_vec),
                "angular_velocity": list(ego_angular_velocity_vec),
                "extra": {} #da aggiungere successivamente
            },

            "other_actors": {},
        }

        # per ego
        ego_bb = ego_vehicle.bounding_box
        ego_vertices: List[carla.Location] = ego_bb.get_world_vertices(ego_vehicle.get_transform())
        ego_vertices_list = [[v.x, v.y, v.z] for v in ego_vertices]

        frame_data["ego_vehicle"]["bounding_box_vertices"] = ego_vertices_list

        for actor_snap in snapshot:
            if not isinstance(actor_snap, carla.ActorSnapshot):
                continue

            actor = world.get_actor(actor_snap.id)
            if actor is None or not actor.is_alive or actor.id == ego_vehicle.id:
                continue

            tf = actor.get_transform()
            loc = actor.get_location()
            vel = actor.get_velocity()
            ang = actor.get_angular_velocity()

            # Se non ancora registrato, prova a registrarlo
            if str(actor_snap.id) not in self.scenario_data["actors"]:
                registered = self.register_actor(
                    actor=actor,
                    actor_role="npc",
                    spawn_transform=[loc.x, loc.y, loc.z,
                                    tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw],
                    snapshot=snapshot
                )
                if not registered:
                    continue  # scarta non dinamici

            # Qui sempre: aggiorna i dati di frame per i dinamici
            frame_data["other_actors"][actor.id] = {
                "type_id": actor.type_id,
                "location": [loc.x, loc.y, loc.z],
                "rotation": [tf.rotation.roll, tf.rotation.pitch, tf.rotation.yaw],
                "velocity": [vel.x, vel.y, vel.z],
                "angular_velocity": [ang.x, ang.y, ang.z],
                "extra": {}
            }

            if not isinstance(actor, (carla.Vehicle, carla.Walker)):
                continue
            # per altri attori dinamici
            bb = actor.bounding_box
            vertices = bb.get_world_vertices(actor.get_transform())
            vertices_list = [[v.x, v.y, v.z] for v in vertices]

            frame_data["other_actors"][actor.id]["bounding_box_vertices"] = vertices_list
        self.scenario_data["frames"].append(frame_data)
        self.frame_count += 1

    def finalize_and_save(self, filename: Optional[str] = None) -> str:
        """
            Finalizza lo scenario (senza calcoli di metriche) e salva il file json
        """
         # Ferma la registrazione del recorder
        if self.record_binary and self.client:
            try:
                self.client.stop_recorder()
                print(f"[CarlaBasicLogger] Binary recording stopped: {self.recorder_path}")

                if self.recorder_path and os.path.exists(self.recorder_path):
                    print(f"[CarlaBasicLogger] File di recording salvato: {self.recorder_path}")
                else:
                    print(f"[CarlaBasicLogger] ATTENZIONE: file non trovato: {self.recorder_path}")
            except Exception as e:
                print(f"[CarlaBasicLogger] Errore nello stop del recorder: {e}")

        self.scenario_data["results"]["total_frames"] = self.frame_count

        # Timestamp real-time
        sim_end_time = time.time()
        self.scenario_data["results"]["total_simulation_time"] = sim_end_time - self.simulation_start_time

        # Timestamp da simulator
        start_ts = self.scenario_data["frames"][0]["timestamp"]
        end_ts = self.scenario_data["frames"][-1]["timestamp"]
        self.scenario_data["results"]["simulated_time"] = end_ts - start_ts

        if filename is None:
            if self.tool == "ScenarioFuzzLLM":
                filename = f"{self.scenario_id}_run_{self.run_index:02d}_log_basic.json"
            else:
                filename = f"{self.generation_id}_{self.scenario_id}_log_basic.json"
        path = os.path.join(self.output_dir, filename)

        rounded_data = round_floats(self.scenario_data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(rounded_data, f, indent=2, ensure_ascii=False)
        print(f"[CarlaBasicLogger] Dati scenario salvati in: {path}")
        self.ended = True
        return path
    

    def log_collision(self, collision_data):
        """Registra le collisioni"""
        self.scenario_data["events"]["collision"].append(collision_data)
        self.scenario_data["results"]["has_collision"] = True

        # contatore totale
        self.scenario_data["results"]["event_counts"]["collision"] += 1

        # contatore specifico per categoria
        cat = collision_data.get("collision_category")
        if cat and cat in self.scenario_data["results"]["event_counts"]:
            self.scenario_data["results"]["event_counts"][cat] += 1


    def log_red_light(self, frame: int, timestamp: float, speed_kmh):
        """Registra un passaggio col rosso"""
        if self.ended:
            return
        self.scenario_data["events"]["red_lights"].append({
            "frame": frame,
            "timestamp": timestamp,
            "speed_kmh": speed_kmh
        })
        print(f"[ViolationMonitor] Red light violation at frame {frame}, timestamp {timestamp}")
        self.scenario_data["results"]["has_red_light_violation"] = True
        self.scenario_data["results"]["event_counts"]["red_light"] += 1

    def log_speeding(self, frame: int, timestamp: float, speed_kmh: float, speed_limit_kmh: float):
        """Registra uno sforamento velocità (grezzo)."""
        if self.ended:
            return
        self.scenario_data["events"]["speeding"].append({
            "frame": int(frame),
            "timestamp": float(timestamp),
            "speed_kmh": float(speed_kmh),
            "speed_limit_kmh": float(speed_limit_kmh)
        })
        print(f"[ViolationMonitor] Speeding violation at frame {frame}, timestamp {timestamp}: ego going {speed_kmh} on a {speed_limit_kmh} limit route")
        self.scenario_data["results"]["has_speeding"] = True
        self.scenario_data["results"]["event_counts"]["speeding"] += 1
    
    def log_stop_violation(self, frame, timestamp, lm_id, lm_loc, speed_kmh, stopped):
        if self.ended:
            return
        violation = {
            "frame": frame,
            "timestamp": timestamp,
            "landmark_id": lm_id,
            "location": [lm_loc.x, lm_loc.y, lm_loc.z],
            "speed_kmh": speed_kmh,
            "stopped": stopped  # True se si è fermato ma troppo poco, False se mai fermato
        }
        self.scenario_data["events"]["stop_sign"].append(violation)

        status = "TOO SHORT" if stopped else "NOT STOPPED"
        print(f"[ViolationMonitor] Stop sign violation ({status}) at frame {frame}, timestamp {timestamp} (ID={lm_id})")
        self.scenario_data["results"]["has_stop_violation"] = True
        self.scenario_data["results"]["event_counts"]["stop_sign"] += 1

    def log_lane_invasion(self, frame: int, timestamp: float, crossed_markings):
        if self.ended:
            return

        self.scenario_data["events"]["lane_invasion"].append({
            "frame": frame,
            "timestamp": timestamp,
            "crossed_markings": crossed_markings
        })
        self.scenario_data["results"]["event_counts"]["lane_invasion"] += 1

    def log_off_road(self, frame: int, timestamp: float, location):
        if self.ended:
            return

        self.scenario_data["events"]["off_road"].append({
            "frame": frame,
            "timestamp": timestamp,
            "location": [location.x, location.y, location.z],
        })

        self.scenario_data["results"]["event_counts"]["off_road"] += 1


    @staticmethod
    def handle_collision(event: carla.CollisionEvent, state):
        logger = LOGGER_REGISTRY.get(event.actor.id)
        if not logger or logger.ended:
            return

        # --------------------------------------------------
        # 1. Evita collisioni multiple con lo stesso attore
        # --------------------------------------------------
        if logger.last_collision_actor_id == event.other_actor.id:
            return

        # --------------------------------------------------
        # 2. Classificazione tipo collisione (Leaderboard)
        # --------------------------------------------------
        if isinstance(event.other_actor, carla.Vehicle):
            collision_category = "collision_vehicle"
        elif isinstance(event.other_actor, carla.Walker):
            collision_category = "collision_pedestrian"
        else:
            collision_category = "collision_static"

        # --------------------------------------------------
        # 3. Aggiorna stato specifico del tool (se serve)
        # --------------------------------------------------
        if logger.tool == "SimADFuzz":
            CarlaBasicLogger.handle_simadfuzz_state(event, state)

        elif logger.tool == "TMFuzzer":
            CarlaBasicLogger.handle_tmfuzz_state(event, state)

        elif logger.tool == "ScenarioFuzzLLM":
            CarlaBasicLogger.handle_scenariofuzzllm_state(event, state)

        # --------------------------------------------------
        # 4. Costruzione dati collisione (coerenti!)
        # --------------------------------------------------
        collision_data = analyze_collision(event=event)

        collision_data["collision_category"] = collision_category
        collision_data["other_actor_id"] = event.other_actor.id
        collision_data["other_actor_type_id"] = event.other_actor.type_id

        # frame e timestamp COERENTI col logger
        collision_data["frame"] = logger.frame_count

        # timestamp relativo alla simulazione
        if logger._first_timestamp is not None:
            collision_data["timestamp"] = (
                logger.world.get_snapshot().timestamp.elapsed_seconds
                - logger._first_timestamp
            )
        else:
            collision_data["timestamp"] = 0.0

        print(
            f"[HAZARD] Collision ({collision_category}) with "
            f"{event.other_actor.type_id} (ID {event.other_actor.id})"
        )

        # --------------------------------------------------
        # 5. Log collisione
        # --------------------------------------------------
        logger.log_collision(collision_data)

        # --------------------------------------------------
        # 6. Aggiorna ultimo attore coinvolto
        # --------------------------------------------------
        logger.last_collision_actor_id = event.other_actor.id



    @staticmethod
    def handle_simadfuzz_state(event:carla.CollisionEvent,state):
        # Early-exit se lo state ha già early_stop
        if state.early_stop:
            return

        state.crashed = True
        state.early_stop = True
        state.early_stop_reason = "Ego collision"
        state.violation_found = True
        state.collision_details.append((event.timestamp, event.transform))


    @staticmethod
    def handle_scenariofuzzllm_state(event: carla.CollisionEvent, state):
        if state.end:
            # ignore collision happened AFTER simulation ends
            # (can happen because of sluggish garbage collection of Carla)
            return
        if event.other_actor.type_id != "static.road":
            if not state.crashed:
                print("COLLISION:", event.other_actor.type_id)
                # do not count collision while spawning ego vehicle (hard drop)
                state.crashed = True
                state.collision_to = event.other_actor.id
                state.min_dist = 0
                state.min_dist_frame = state.num_frames

    @staticmethod
    def handle_tmfuzz_state(event:carla.CollisionEvent,state):
        if state.end:
            # ignore collision happened AFTER simulation ends
            # (can happen because of sluggish garbage collection of Carla)
            return
        if event.other_actor.type_id != "static.road":
            if not state.crashed:
                print("COLLISION:", event.other_actor.type_id)
                # do not count collision while spawning ego vehicle (hard drop)
                state.crashed = True
                state.collision_to = event.other_actor.id


def round_floats(data: Any, decimals=4) -> Any:
    if isinstance(data, float):
        return round(data, decimals)
    elif isinstance(data, list):
        return [round_floats(x, decimals) for x in data]
    elif isinstance(data, dict):
        return {k: round_floats(v, decimals) for k, v in data.items()}
    else:
        return data