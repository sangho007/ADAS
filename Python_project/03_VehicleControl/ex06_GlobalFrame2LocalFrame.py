import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

class Global2Local(object):
    def __init__(self, num_points):
        self.n = num_points
        self.GlobalPoints = np.zeros((num_points,2))
        self.LocalPoints = np.zeros((num_points,2))
    
    def convert(self, points, Yaw_ego, X_ego, Y_ego):
        # Code
        self.GlobalPoints = points

        # 변환 행렬 (회전만 포함, 평행 이동은 점에 직접 적용)
        self.TransMatrix = np.array([
            [np.cos(Yaw_ego), np.sin(Yaw_ego)],
            [-np.sin(Yaw_ego), np.cos(Yaw_ego)],
        ])

        for i, point in enumerate(self.GlobalPoints):
            # 전역 좌표계에서의 평행 이동 적용
            translated_point = np.array([point[0] - X_ego, point[1] - Y_ego])

            # 변환 행렬 적용
            P_dot = self.TransMatrix @ translated_point

            # 변환된 점 저장 (Z 좌표는 무시)
            self.LocalPoints[i, :] = P_dot


class PolynomialFitting(object):
    def __init__(self, num_degree, num_points):
        self.nd = num_degree
        self.np = num_points
        self.A = np.zeros((self.np, self.nd + 1))
        self.b = np.zeros((self.np, 1))
        self.coeff = np.zeros((num_degree + 1, 1))

    def fit(self, points):
        points = np.array(points)  # points가 리스트라면 NumPy 배열로 변환
        for i, p in enumerate(points):
            for j in range(self.nd + 1):
                self.A[i, j] = p[0] ** (self.nd - j)
        self.b = points[:, 1].reshape(-1, 1)  # b 벡터를 y값으로 채움
        self.coeff = np.linalg.pinv(self.A.T @ self.A) @ self.A.T @ self.b


class PolynomialValue(object):
    def __init__(self, num_degree, num_points):
        self.nd = num_degree  # 다항식의 차수
        self.np = num_points  # 포인트의 수
        self.points = np.zeros((self.np, 2))  # x와 y값 쌍을 저장할 2차원 배열


    def calculate(self, coeff, x_values):
        # x_values에 대한 y값을 계산하고 저장할 리스트
        y_values = []

        for x in x_values:
            y = 0
            for i, c in enumerate(coeff):
                # c: 계수, i: 인덱스
                # x의 i번째 거듭제곱과 계수를 곱하여 y에 더함
                y += c * (x ** (len(coeff) - 1 - i))
            y_values.append(y)

        # 계산된 y값들을 numpy 배열로 변환
        y_values = np.array(y_values).reshape(-1, 1)  # y_values를 열 벡터로 변환

        # self.points 배열에 x값과 y값 쌍을 저장
        self.points = np.hstack((x_values.reshape(-1, 1), y_values))  # x_values와 y_values를 옆으로 결합




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