import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from shapely.geometry import Polygon


def polygon_from_vertices(vertices:List) -> Optional[Polygon]:
    """
    Converte una lista di vertici 3D in un poligono 2D valido 
    per calcolare la distanza tra attori in maniera più precisa.
    """
    #conversione da 3d a 2d per semplificare i calcoli
    points_2d = [(float(vertex[0]), float(vertex[1])) for vertex in vertices]

    #eliminazione di vertici consecutivi identici
    unic_points = []
    for index, point in enumerate(points_2d):
        if index == 0 or point != points_2d[index -1]:
            unic_points.append(point)

    if len(unic_points) < 3:
        return None
    
    #creazione del poligono iniziale
    polygon = Polygon(unic_points)

    #se il poligono è malformato
    if not polygon.is_valid:
        #cerca di renderlo valido rimuovendo anomalie
        polygon = polygon.buffer(0)

    #conversione del poligono ad un convex hull
    convex_hull = polygon.convex_hull

    return convex_hull if isinstance(convex_hull, Polygon) else None

def relative_speed_magnitude(ego_velocity: List, other_velocity: List) -> float:
    """
    Calcola il modulo della velocità relativa tra due attori (in 2D)

    Velocità relativa = velocità_ego - velocità_altro_attore
    """
    #calcolo componenti velocità relativa nel piano XY
    delta_vx = float(ego_velocity[0] - other_velocity[0])
    delta_vy = float(ego_velocity[1] - other_velocity[1])

    #modulo del vettore della velocità relativa
    return math.hypot(delta_vx, delta_vy)

def safe_polygon_distance(polygon_a: Polygon, polygon_b: Polygon):
    """
    Calcola la distanza 2D minima tra due poligoni.
    
    Logica di calcolo:
    1. Se i poligoni si intersecano: distanza = 0 (situazione critica)
    2. Se separati: distanza euclidea minima tra bordi più vicini
    """
    return 0.0 if polygon_a.intersects(polygon_b) else polygon_a.distance(polygon_b)

@dataclass
class CriticalAnalyzer:
    """
    Aggregatore per il calcolo delle metriche di criticità.
    
    Questo sistema mantiene lo stato cumulativo durante l'analisi frame-by-frame di uno scenario,
    tracciando contemporaneamente:
    
    1. Minimum distance:
       - Distanza minima globale in tutto lo scenario
       - Dettagli dell'evento: frame, attore coinvolto
       - Tracking per singolo attore: MDBV individuale per ogni attore dinamico
    
    2. min_TTC (minimum Time To Collision):
       - TTC minimo globale registrato
       - Metadati: momento temporale, attore responsabile
    
    3. TET (Time-Exposed Time-to-Collision):
       - Tempo totale sotto soglia TTC critica
       - Sequenza continua massima (worst-case exposure)
       - Tracking delle finestre temporali di rischio
    """

    # ==========
    # Parametri di configurazione
    # ==========
    ttc_threshold: float = 1.5 #Soglia TTC critica sotto la quale calcolare TET
    delta_time: float = 0.05   #Durata temporale di ogni frame

    # ==========
    # Stato MDBV 
    # ==========
    min_distance: float = float("inf") #distanza minima tra da ogni attore
    min_distance_frame: int = -1       #indice del frame in cui è stata registrata la distanza minima
    min_distance_actor: Dict[str, Any] = field(default_factory=dict) #dettagli attore: {id, type_id}

    #tracking distanza per singolo attore - struttura: {actor_id : {type_id, min_distance, frame}}
    min_distance_per_actor: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ==========
    # Stato TTC 
    # ==========
    min_ttc: float = float("inf") #ttc minimo 
    min_ttc_frame: int = -1       # indice del frmae in cui è stato registrato il ttc minimo
    min_ttc_actor: Dict[str, Any] = field(default_factory=dict) #attore responsabile del ttc minimo

    # ==========
    # Stato TET
    # ==========

    #accumulo temporale totale: somma di tutti i periodi sotto ttc_threshold
    total_exposure: float = 0.0

    #gestione sequenze continue
    current_streak: float = 0.0 #durata sequenza attuale sotto threshold
    max_streak: float = 0.0     #durata massima sequenza mai registrata

    #dettagli sequenza massima
    max_streak_start_frame: Optional[int] = None 
    max_streak_end_frame: Optional[int] = None

    #tracking sequenza attuale
    current_streak_start_frame: Optional[int] = None

    def update(self, frame_index: int, frame_min_tcc: float, frame_metrics: List[Dict]):
        """
        Aggiorna lo stato aggregato con i dati di un nuovo frame analizzato.

        Operazioni eseguite:
        1. Aggiornamento MDBV globale e per singolo attore
        2. Aggiornamento min_TTC globale con identificazione attore responsabile
        3. Progressione TET

        Logica TET:
            - TTC <= soglia: estende sequenza corrente o ne inizia una nuova
            - TTC > soglia: chiude sequenza corrente e aggiorna statistiche
        """

        # ==========
        # aggiornamento MDBV 
        # ==========

        for metric in frame_metrics:
            actor_id = metric["actor_id"]
            current_distance = metric["distance"]
            actor_type = metric["actor_type"]

            #verifica se questa è la nuova distanza minima
            if current_distance < self.min_distance:
                self.min_distance = current_distance
                self.min_distance_frame = frame_index
                self.min_distance_actor = {
                    "id": actor_id,
                    "type_id": actor_type
                }
            
            #aggiornamento MDBV per questo specifico attore
            if (actor_id not in self.min_distance_per_actor or
                current_distance < self.min_distance_per_actor[actor_id]["min_distance"]):

                self.min_distance_per_actor[actor_id] = {
                    "type_id": actor_type,
                    "min_distance": current_distance,
                    "frame": frame_index
                }

        # ==========
        # aggiornamento TTC 
        # ==========

        if frame_min_tcc < self.min_ttc:
            self.min_ttc = frame_min_tcc
            self.min_ttc_frame = frame_index

            self.min_ttc_actor = {}
            for metric in frame_metrics:
                if metric.get("ttc") == frame_min_tcc:
                    self.min_ttc_actor = {
                        "id": metric["actor_id"],
                        "type_id": metric["actor_type"]
                    }

        # ==========
        # aggiornamento TET 
        # ==========

        if frame_min_tcc <= self.ttc_threshold:
            #inizio nuova sequenza di esposizine
            if self.current_streak == 0.0:
                self.current_streak_start_frame = frame_index

            #estensione sequenza attuale
            self.current_streak += self.delta_time
        else:
            #chiusura eventuale sequenza in corso
            self.end_streak(frame_index)

    def end_streak(self, current_frame_index: int):
        """
        Termina una sequenza di esposizione continua e aggiorna le statistiche TET.
        
        Questo metodo viene chiamato quando:
        1. Il TTC supera la soglia critica (fine situazione rischiosa)
        2. Si raggiunge la fine dello scenario (finalizzazione)
        """
         
        #verifica se c'è una sequena attiva da chiudere
        if self.current_streak > 0:
            #accumula nel tempo totale
            self.total_exposure += self.current_streak

            #verifica se questa sequenza è la più lunga finora registrata
            if self.current_streak > self.max_streak:
                self.max_streak = self.current_streak
                self.max_streak_start_frame = self.current_streak_start_frame
                #la sequenza termina al frame antecedente a quello attuale
                self.max_streak_end_frame = current_frame_index - 1

            #reset per la prossima sequenza
            self.current_streak = 0.0
            self.current_streak_start_frame = None

    def get_results(self) -> Dict[str, Any]:
        """
        Estrae il risultato finale completo dell'analisi di criticità
        assemblando tutte le metriche calcolate in un dizionario.
        """

        # Converti il dizionario in una lista ordinata per MDBV crescente
        actors_ordered_by_distance = []
        if self.min_distance_per_actor:
            # Ordina per distanza minima crescente
            sorted_actors = sorted(
                self.min_distance_per_actor.items(),
                key=lambda element: element[1]["min_distance"]
            )
        
        for actor_id, data in sorted_actors:
            actors_ordered_by_distance.append({
                "actor_id": actor_id,
                "type_id": data["type_id"], 
                "min_distance": data["min_distance"],
                "frame": data["frame"]
            })

        return {
            # Metriche MDBV 
            "MDBV": self.min_distance if self.min_distance != float("inf") else None,
            "MDBV_frame": self.min_distance_frame,
            "MDBV_actor": self.min_distance_actor,
            
            # Metriche min_TTC (minimum Time To Collision)
            "min_TTC": self.min_ttc if self.min_ttc != float("inf") else None,
            "min_TTC_frame": self.min_ttc_frame,
            "min_TTC_actor": self.min_ttc_actor,
            
            # Metriche TET (Total Exposure Time)
            "TET_total": round(self.total_exposure, 3),      
            "TET_max": round(self.max_streak, 3),            
            "TET_max_start_frame": self.max_streak_start_frame,
            "TET_max_end_frame": self.max_streak_end_frame,
            
            # Analisi dettagliata per attore
            "MDBV_per_actor": actors_ordered_by_distance,
        }
    
def calculate_scenario_metrics(frames: List[Dict], ttc_threshold: float = 1.5, delta_time: float = 0.05):
    """
    Funzione orchestratrice - Analizza una sequenza di frame 
    per calcolare le metriche di criticità.
    """ 

    if not frames:
        #input vuoto
        return {}
    
    aggregator = CriticalAnalyzer(ttc_threshold=ttc_threshold, delta_time=delta_time)
    processed_frames = 0

    for frame_index, frame in enumerate(frames):
        #1. Estrazione e validazione geometria dell'ego-vehicle
        ego_data = frame["ego_vehicle"]
        ego_polygon = polygon_from_vertices(ego_data["bounding_box_vertices"])

        if ego_polygon is None:
            logging.warning(
                f"Frame {frame_index}: impossibile creare il poligono per l'ego vehicle. "
                f"Salto il frame."
            )
            continue
        
        ego_velocity = ego_data.get("velocity", [0, 0, 0])

        #2. Preparazione strutture dati per frame attuale
        frame_metrics: List[Dict] = []
        min_ttc_frame = float("inf")

        other_actors = frame.get("other_actors", {})

        if not other_actors:
            continue

        processed_frames += 1

        #3. Analisi coppia ego-attore per ogni altro attore dinamico
        for actor_id, actor_data in other_actors.items():
            actor_polygon = polygon_from_vertices(actor_data["bounding_box_vertices"])
            if actor_polygon is None:
                continue

            distance = safe_polygon_distance(ego_polygon, actor_polygon)

            actor_velocity = actor_data.get("velocity", [0, 0, 0])
            relative_speed = relative_speed_magnitude(ego_velocity, actor_velocity)

            if relative_speed > 1e-6:
                ttc = distance / relative_speed
            else:
                ttc = float("inf")

            min_ttc_frame = min(min_ttc_frame, ttc)

            actor_metric = {
                "actor_id": actor_id,
                "actor_type": actor_data["type_id"],
                "distance": distance,
                "ttc": ttc
            }
            frame_metrics.append(actor_metric)

        aggregator.update(frame_index, min_ttc_frame, frame_metrics)

    aggregator.end_streak(processed_frames)
    return aggregator.get_results()

