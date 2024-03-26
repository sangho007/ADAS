import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue

def polyval(coeff, x):
    x_matrix = np.zeros((1,np.size(coeff)))
    for i in range(np.size(coeff)):
        x_matrix[0][i]=(x**(np.size(coeff)-1-i))
    y = x_matrix@coeff
    return y[0][0]

class PID_Controller_Kinematic(object):
    def __init__(self, step_time, coeff, Vx, lookahead_time = 0.5, kp=0.2, kd=0.1, ki=0.0, kff = 0.4):
        self.dt = step_time
        self.Kp = kp
        self.Kd = kd
        self.Ki = ki
        self.kff = kff
        self.t_lh = lookahead_time
        self.d_lh = self.t_lh * Vx
        self.error = polyval(coeff, self.d_lh)
        self.error_prev = self.error
        self.error_i = 0.0
        self.max_delta_error = 3.0
        
    def ControllerInput(self, coeff, Vx):
        self.d_lh = self.t_lh * Vx
        self.error = polyval(coeff, self.d_lh)
        self.error_d = np.min([(self.error - self.error_prev),self.max_delta_error])/self.dt
        self.error_i = self.error_i + self.error * self.dt
        self.feedforwardterm = Vx**2 * 2*coeff[-3][0]
        self.u = self.Kp * self.error + self.Kd * self.error_d + self.Ki * self.error_i + self.kff * self.feedforwardterm
        self.error_prev = self.error

    
if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_ref = np.arange(0.0, 100.0, 0.1)
    Y_ref = 2.0-2*np.cos(X_ref/10)
    num_degree = 3
    num_point = 5
    x_local = np.arange(0.0, 10.0, 0.5)
    error_rms = 0.0
    err_i = 0

    time = []
    X_ego = []
    Y_ego = []
    ego_vehicle = VehicleModel_Lat(step_time, Vx)

    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    controller = PID_Controller_Kinematic(step_time, polynomialfit.coeff, Vx)
    
    for i in range(int(simulation_time/step_time)):
        time.append(step_time*i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        X_ref_convert = np.arange(ego_vehicle.X, ego_vehicle.X+5.0, 1.0)
        Y_ref_convert = 2.0-2*np.cos(X_ref_convert/10)
        Points_ref = np.transpose(np.array([X_ref_convert, Y_ref_convert]))
        frameconverter.convert(Points_ref, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        polynomialfit.fit(frameconverter.LocalPoints)
        polynomialvalue.calculate(polynomialfit.coeff, x_local)
        controller.ControllerInput(polynomialfit.coeff, Vx)
        ego_vehicle.update(controller.u, Vx)
        error_rms = error_rms + np.sqrt((ego_vehicle.Y-Y_ref_convert[0])**2)
        err_i = err_i+1

    print(error_rms/err_i)
    plt.figure(1)
    plt.plot(X_ref, Y_ref,'k-',label = "Reference")
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    #plt.axis("equal")
    plt.grid(True)    
    plt.show()


