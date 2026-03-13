import carla

from .behavior_agent import BehaviorAgent
from .stop_manager import StopManager


class CustomBehaviorAgent(BehaviorAgent):
    def __init__(self, vehicle: carla.Vehicle, behavior='custom'):
        super().__init__(vehicle=vehicle, behavior=behavior)

        #istanza dello stop_manager
        self.stop_manager = StopManager(vehicle=self._vehicle, world=self._world, map=self._map)

    def run_step(self, debug=False):
        """
        Override del metodo run_step per integrare la logica dello StopManager
        """
        self._update_information()

        #controllo dello stop sign come priorità massima
        if self.stop_manager.update():
            #se ritorna True, bisogna frenare
            return super().emergency_stop()
        
        return super().run_step(debug=debug)
        
        