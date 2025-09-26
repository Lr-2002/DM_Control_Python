import numpy 
class StaticGravityCompensation:
    def __init__(self):
        self.torque = np.array([4, 0, 4, 0, 1, 0, 0, 0, 0])
        pass 

    def calculate_torque(self, q, dq, ddq):
        return self.torque

    def calculate_gravity_torque(self, q):
        
        return self.torque
    
    def calculate_coriolis_torque(self, q, dq):
        return np.zeros(9)
    