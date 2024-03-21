import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average=15):
        self.y_estimate = y_initial_measure
        self.data = y_initial_measure*np.ones((num_average, 1))
        self.num_average = num_average
        
    def estimate(self, y_measure):
        self.y_estimate = self.y_estimate + (y_measure-self.data[-1])/self.num_average
        self.data = np.roll(self.data,1)
        self.data[0] = y_measure

    
if __name__ == "__main__":
    #signal = pd.read_csv("01_Filter/Data/example_Filter_1.csv")      
    signal = pd.read_csv("01_Filter/Data/example_Filter_2.csv")

    y_estimate = MovingAverageFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i])
        signal.y_estimate[i] = y_estimate.y_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



