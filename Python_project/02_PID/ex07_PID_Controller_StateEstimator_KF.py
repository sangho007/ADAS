from vehicle_model import VehicleModel
import numpy as np
import matplotlib.pyplot as plt

class PID_Controller(object):
    def __init__(self, reference, measure, step_time, P_Gain=0.5, D_Gain=1.2, I_Gain=0.02):
        self.Kp = P_Gain
        self.Kd = D_Gain
        self.Ki = I_Gain
        self.dt = step_time
        self.error_prev = reference - measure
        self.error_i = 0
        self.u = 0.0
    
    def ControllerInput(self, reference, measure):
        self.error = reference - measure
        self.error_d = (self.error - self.error_prev)/self.dt
        self.error_i = self.error_i + self.error * self.dt
        self.u = self.Kp * self.error + self.Kd * self.error_d + self.Ki * self.error_i
        self.error_prev = self.error
        
class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.1, m=1.0, Q_x=0.01, Q_v=0.01, R=2.0, errorCovariance_init=10.0):
        self.A = np.array([[1.0, step_time], [0.0, 1.0]])
        self.B = np.array([[0.0], [step_time/m]])
        self.C = np.array([[1.0, 0.0]])
        self.D = 0.0
        self.Q = np.array([[Q_x, 0.0], [0.0, Q_v]])
        self.R = R
        self.x_estimate = np.array([[y_Measure_init],[0.0]])
        self.P_estimate = np.array([[errorCovariance_init, 0.0],[0.0, errorCovariance_init]])

    def estimate(self, y_measure, input_u):
        # Prediction
        self.x_prediction = self.A @ self.x_estimate  +  self.B * input_u
        self.P_prediction = (self.A) @ self.P_estimate @ (self.A.T)  +  self.Q
        # Update
        self.kalman_gain = (self.P_prediction@self.C.T)/((self.C@self.P_prediction@self.C.T)+self.R)
        self.x_estimate = self.x_prediction  +  self.kalman_gain * (y_measure - self.C@self.x_prediction)
        self.P_estimate = (np.identity(2) - self.kalman_gain@self.C)@self.P_prediction

if __name__ == "__main__":
    target_y = 0.0
    measure_y =[]
    estimated_y = []
    time = []
    step_time = 0.1
    simulation_time = 30   
    plant = VehicleModel(step_time, 0.25, 0.99, 0.05)
    estimator = KalmanFilter(plant.y_measure[0][0])
    controller = PID_Controller(target_y, plant.y_measure[0][0], step_time)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        measure_y.append(plant.y_measure[0][0])
        estimated_y.append(estimator.x_estimate[0][0])
        estimator.estimate(plant.y_measure[0][0],controller.u)
        controller.ControllerInput(target_y, estimator.x_estimate[0][0])
        plant.ControlInput(controller.u)
    
    plt.figure()
    plt.plot([0, time[-1]], [target_y, target_y], 'k-', label="reference")
    plt.plot(time, measure_y,'r:',label = "Vehicle Position(Measure)")
    plt.plot(time, estimated_y,'c-',label = "Vehicle Position(Estimator)")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()
