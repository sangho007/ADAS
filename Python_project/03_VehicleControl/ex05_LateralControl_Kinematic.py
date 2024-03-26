import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat

class PID_Controller_Kinematic(object):
    def __init__(self, step_time, reference_Y, ego_Y, kp=0.2, kd=0.8, ki=0.0):
        self.dt = step_time
        self.Kp = kp
        self.Kd = kd
        self.Ki = ki
        self.error = reference_Y - ego_Y
        self.error_prev = self.error
        self.max_delta_error = 3.0
        self.error_i = 0
    def ControllerInput(self, reference_Y, ego_Y):
        self.error = reference_Y - ego_Y
        self.error_d = np.min([(self.error - self.error_prev),self.max_delta_error])/self.dt
        self.error_i = self.error_i + self.error * self.dt
        self.u = self.Kp * self.error + self.Kd * self.error_d + self.Ki * self.error_i
        self.error_prev = self.error

    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    Y_ref = 4.0
    
    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PID_Controller_Kinematic(step_time, Y_ref, ego_vehicle.Y)
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        controller.ControllerInput(Y_ref, ego_vehicle.Y)
        ego_vehicle.update(controller.u, Vx)

        
    plt.figure(1)
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.plot([0, X_ego[-1]], [Y_ref, Y_ref], 'k:',label = "Reference")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


