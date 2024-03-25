import numpy as np
import matplotlib.pyplot as plt

from VehicleModel_Lat import VehicleModel_Lat
from ex06_GlobalFrame2LocalFrame import Global2Local
from ex06_GlobalFrame2LocalFrame import PolynomialFitting
from ex06_GlobalFrame2LocalFrame import PolynomialValue

def polyval(coeff, x): # 다항식 값 구하는 함수
    x_matrix = np.zeros((1, np.size(coeff)))
    for i in range(np.size(coeff)):
        x_matrix[0][i] = (x**(np.size(coeff)-1-i))
    y = x_matrix@coeff
    return y[0][0]

class PID_Controller_Kinematic(object):
    def __init__(self, step_time, coeff, Vx, Kp=1.0, Ki=0, Kd=0):
        # Code
        self.dt = step_time
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki

        self.look_aheadTime = 1.0
        self.d_l = Vx * self.look_aheadTime # loot ahead distance

        self.error = polyval(coeff, self.d_l)   # d_l을 구해서 polynomial 다항식에 집어 넣은게 곧 에러이다.
        self.error_prev = self.error

        self.max_delta_error = 3.0      # 최대 에러값
        self.error_i = 0

    def ControllerInput(self, coeff, Vx):
        # Code
        self.d_l = Vx * self.look_aheadTime
        self.error = self.error = polyval(coeff, self.d_l)

        self.error_d = np.min([(self.error - self.error_prev), self.max_delta_error]) / self.dt
        self.error_i = self.error_i + self.error * self.dt

        self.u = self.Kp * self.error + self.Kd * self.error_d + self.Ki * self.error_i

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

    plt.figure(1)
    plt.plot(X_ref, Y_ref,'k-',label = "Reference")
    plt.plot(X_ego, Y_ego,'b-',label = "Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
#    plt.axis("best")
    plt.grid(True)    
    plt.show()


