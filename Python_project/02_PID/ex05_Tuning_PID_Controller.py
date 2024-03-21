from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=5.0, D_Gain=5.0, I_Gain=0.35):
        # Code
        self.Kp = P_Gain
        self.Ki = I_Gain
        self.Kd = D_Gain
        self.step_time = step_time
        self.error_old = (reference - measure)
        self.s_error = 0
        self.u = 0.0

    def ControllerInput(self, reference, measure):
        # Code
        self.error = reference - measure
        self.d_error = (self.error - self.error_old) / self.step_time
        self.s_error += self.error * self.step_time  # integral 이므로 step 곱해주자
        self.u = self.Kp * self.error + self.Kd * self.d_error + self.Ki * self.s_error
        self.error_old = self.error


if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.0, 0.4, -0.1)
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        controller.ControllerInput(target_y, plant.y_measure[0][0])
        plant.ControlInput(controller.u)

    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r-',label = "Vehicle Position")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
