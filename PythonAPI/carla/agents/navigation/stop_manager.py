import carla
from typing import List, Optional


class StopManager:
    PROXIMITY_THRESHOLD = 3.0 #m
    SPEED_THRESHOLD = 0.3 #m/s
    WAYPOINT_STEP = 1.0 #m
    STOP_DURATION = 2.0 #s

    def __init__(self, vehicle: carla.Vehicle, world: carla.World, map:carla.Map):
        self.vehicle = vehicle
        self.world = world
        self.map = map

        self.stop_signs: List[carla.TrafficSign] = self.world.get_actors().filter("*stop*")

        self.affected_by_stop = False
        self.stop_completed = False
        self.target_stop_sign: Optional[carla.TrafficSign] = None
        self.stop_start_time = None

        #Disegna le bounding box dei segnali di stop, per debug
        '''for stop in self.stop_signs:
            stop_bb:carla.BoundingBox = stop.trigger_volume
            stop_transform = stop.get_transform()
            world_bb = carla.BoundingBox(stop_transform.transform(stop_bb.location), stop_bb.extent)

            self.world.debug.draw_box(
                box = world_bb,
                rotation = stop_transform.rotation,
                thickness = 0.1,
                color = carla.Color(0, 200, 200, 200),
                life_time = 0 
        )'''


    def update(self) -> bool:
        """
        Chiamato ad ogni tick: ritorna True se bisogna frenare per uno stop sign, False altrimenti
        """
        if self.affected_by_stop:
            return self.handle_current_stop()
        else:
            return self.check_new_stop()
        
    def handle_current_stop(self) -> bool:
        current_speed = self.get_forward_speed()
        now = self.world.get_snapshot().timestamp.elapsed_seconds

        #se non ha completato uno stop
        if not self.stop_completed:
            #se l'attuale velocità è pari a 1 km/h circa
            if current_speed < self.SPEED_THRESHOLD:
                if self.stop_start_time is None:
                    self.stop_start_time = now
                elif now - self.stop_start_time >= self.STOP_DURATION:
                    self.stop_completed = True
                    return False #stop completato, veicolo può ripartire
            return True #applica frenata
        
        else:
            #stop compeltato: verifica se si è allontanato
            if not self.is_actor_affected_by_stop(self.vehicle, self.target_stop_sign):
                #reset
                self.affected_by_stop = False
                self.stop_completed = False
                self.target_stop_sign = None
                self.stop_start_time = None
            return False #veicolo non deve fermarsi

    def check_new_stop(self) -> bool:
        vehicle_transform = self.vehicle.get_transform()
        vehicle_direction = vehicle_transform.get_forward_vector()

        wp: Optional[carla.Waypoint] = self.map.get_waypoint(vehicle_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return False
        
        wp_dir = wp.transform.get_forward_vector()

        #dot_product = veh_dir.x * wp_dir.x + veh_dir.y * wp_dir.y + veh_dir.z * wp_dir.z
        dot_product = vehicle_direction.dot(wp_dir)
        if dot_product <= 0:
            return False #contromano o in direzione errata
        
        for stop_sign in self.stop_signs:
            if self.is_actor_affected_by_stop(self.vehicle, stop_sign):
                self.affected_by_stop = True
                self.target_stop_sign = stop_sign
                return True
            
        return False
    
    def is_actor_affected_by_stop(self, actor: carla.Actor, stop: Optional[carla.TrafficSign], multi_step = 20):
        """
        Verifica se l'attore (in questo caso il veicolo ego) è influenzato dallo stop (entra nel trigger volume)
        """
        if stop is None:
            return False
        
        current_location = actor.get_location()
        stop_location = stop.get_location()
        if stop_location.distance(current_location) > self.PROXIMITY_THRESHOLD:
            return False
        
        #prendi il waypoint corrispondente all'attuale posizione del vicolo ego e verifica per più steps avanti
        wp = self.map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return False
        

        stop_bb: carla.BoundingBox = stop.trigger_volume
        stop_tf = stop.get_transform()

        #ora verifica se uno dei possibili prossimi wp è all'interno del trigger volume dello stop sign
        for _ in range(multi_step):
            if not wp:
                break
            #verifica se è all'internp
            if stop_bb.contains(wp.transform.location, stop_tf):
                return True
            
            #altrimenti prendi i (o il) prossimi waypoint e verifica ugualmente
            next_list = wp.next(self.WAYPOINT_STEP)
            if not next_list:
                break
            wp = next_list[0]

        return False

    def get_forward_speed(self):
        vel = self.vehicle.get_velocity()

        #ottieni vettore direzionale forward del veicolo
        #tale vettore tiene conto di pitch e yaw, e calcola la direzione lungo l'asse longitudinale (x)
        forward_vec = self.vehicle.get_transform().get_forward_vector()

        return vel.dot(forward_vec)