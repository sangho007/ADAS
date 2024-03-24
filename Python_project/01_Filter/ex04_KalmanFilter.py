import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, y_Measure_init, m = 1.0, modelVariance = 0.01, measureVariance = 1.0, errorVariance_init = 10.0):
        self.A = 1.0
        self.B = 1/m
        self.C = 1.0
        self.D = 0.0
        self.Q = modelVariance
        self.R = measureVariance
        self.x_estimate = y_Measure_init
        self.P_estimate = errorVariance_init

    def estimate(self, y_measure, input_u):
        # Prediction
        self.x_prediction = self.A * self.x_estimate  +  self.B * input_u
        self.P_prediction = (self.A)**2 * self.P_estimate  +  self.Q
        # Update
        self.kalman_gain = (self.P_prediction*self.C)/((self.C)**2*self.P_prediction+self.R)
        self.x_estimate = self.x_prediction  +  self.kalman_gain * (y_measure - self.C*self.x_prediction)
        self.P_estimate = (1 - self.kalman_gain*self.C)*self.P_prediction


if __name__ == "__main__":
    signal = pd.read_csv("01_Filter/Data/example_KalmanFilter_1.csv")

    y_estimate = KalmanFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i],signal.u[i])
        signal.y_estimate[i] = y_estimate.x_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



