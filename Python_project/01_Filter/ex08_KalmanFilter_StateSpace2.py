import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, y_Measure_init, step_time=0.1, m=10.0, k=100.0, b=2.0, Q_x=0.01, Q_v=0.1, R=0.1, errorCovariance_init=10.0):
        self.A = np.array([[1.0, step_time], [-step_time*k/m, 1.0-step_time*b/m]])
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
    signal = pd.read_csv("01_Filter/Data/example08.csv")

    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate[0][0]

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    #plt.plot(signal.time, signal.y_true,'b-',label = "True")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



