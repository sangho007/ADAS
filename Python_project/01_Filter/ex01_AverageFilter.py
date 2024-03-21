import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AverageFilter:
    def __init__(self, y_initial_measure):
        self.y_estimate = y_initial_measure
        self.num_data = 0
         
    def estimate(self, y_measure):
        self.y_estimate = self.y_estimate * (self.num_data) / (self.num_data+1) + y_measure / (self.num_data+1)
        self.num_data = self.num_data + 1
    
    
if __name__ == "__main__":
    # signal = pd.read_csv("d:/GitWS/IVS_planning_control/01_Python_project/01_Filter/Data/example_Filter_1.csv")
    signal = pd.read_csv("01_Filter/Data/example_Filter_2.csv")

    y_estimate = AverageFilter(signal.y_measure[0])
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


#for i, row in signal.iterrows():
#    #print(signal.time[i])
#    if (i==0):
#        signal.y_estimate[i] = signal.y_measure[i]
#    else:
#        signal.y_estimate[i] = (signal.y_estimate[i-1])*(num_average-1)/num_average + (signal.y_measure[i])/num_average




