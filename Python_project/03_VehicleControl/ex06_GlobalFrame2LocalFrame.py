import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt
import numpy as np


class Global2Local(object):
    def __init__(self,num_point):
        self.np=num_point
        self.GlobalPoints = None
        self.LocalPoints = None

    def convert(self, points, Yaw_ego, X_ego, Y_ego):
        self.GlobalPoints = points
        num_points_to_convert = min(len(points), self.np)  # Ensure we do not exceed the size of LocalPoints
        self.LocalPoints = np.zeros((num_points_to_convert, 2))  # Adjust the size if necessary

        # Transformation matrix (only rotation, translation is applied directly to the point)
        cos_yaw = np.cos(Yaw_ego)
        sin_yaw = np.sin(Yaw_ego)
        self.TransMatrix = np.array([
            [cos_yaw, sin_yaw],
            [-sin_yaw, cos_yaw],
        ])

        for i, point in enumerate(self.GlobalPoints):
            # Apply global coordinate translation
            translated_point = np.array([point[0] - X_ego, point[1] - Y_ego])

            # Apply transformation matrix
            P_dot = self.TransMatrix @ translated_point

            # Store transformed point (ignoring Z coordinate)
            self.LocalPoints[i, :] = P_dot


class PolynomialFitting(object):
    def __init__(self, num_degree,num_point):
        self.nd = num_degree
        self.np = num_point
        self.coeff = None

    def fit(self, points):
        A = np.zeros((self.np, self.nd + 1))
        b = np.zeros((self.np, 1))

        for i, p in enumerate(points):
            for j in range(self.nd + 1):
                A[i, j] = p[0] ** (self.nd - j)
        b = np.array(points)[:, 1].reshape(-1, 1)  # Fill b vector with y values

        # Perform least squares fitting
        self.coeff = np.linalg.pinv(A.T @ A) @ A.T @ b


class PolynomialValue(object):
    def __init__(self,num_degree,num_point):
        self.nd = num_degree
        self.np = num_point
        self.points = np.zeros((num_point, 2))

    def calculate(self, coeff, x_values):
        y_values = []
        for x in x_values:
            y = sum(c * (x ** (self.nd - i)) for i, c in enumerate(coeff))
            y_values.append(y)

        # Combine x and y values into a single array
        self.points = np.column_stack((x_values, y_values))







if __name__ == "__main__":
    num_degree = 3
    num_point = 4
    points = np.array([[1,2],[3,3],[4,4],[5,5]])
    X_ego = 2.0
    Y_ego = 0.0
    Yaw_ego = np.pi/4
    x_local = np.arange(0.0, 10.0, 0.5)
    
    frameconverter = Global2Local(num_point)
    polynomialfit = PolynomialFitting(num_degree,num_point)
    polynomialvalue = PolynomialValue(num_degree,np.size(x_local))
    frameconverter.convert(points, Yaw_ego, X_ego, Y_ego)
    polynomialfit.fit(frameconverter.LocalPoints)
    polynomialvalue.calculate(polynomialfit.coeff, x_local)
    
    plt.figure(1)
    for i in range(num_point):
        plt.plot(points[i][0],points[i][1],'b.')
    plt.plot(X_ego,Y_ego,'ro',label = "Vehicle")
    plt.plot([X_ego, X_ego+0.2*np.cos(Yaw_ego)],[Y_ego, Y_ego+0.2*np.sin(Yaw_ego)],'r-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.title("Global Frame")
    plt.grid(True)    
    
    plt.figure(2)
    for i in range(num_point):
        plt.plot(frameconverter.LocalPoints[i][0],frameconverter.LocalPoints[i][1],'b.')
    plt.plot(polynomialvalue.points.T[0],polynomialvalue.points.T[1],'b:')
    plt.plot(0.0, 0.0,'ro',label = "Vehicle")
    plt.plot([0.0, 0.5],[0.0, 0.0],'r-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc="best")
    plt.axis((-10,10,-10,10))
    plt.title("Local Frame")
    plt.grid(True)   
    
    plt.show()