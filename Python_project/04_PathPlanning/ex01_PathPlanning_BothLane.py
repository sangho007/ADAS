import numpy as np
import matplotlib.pyplot as plt
from lane_1 import lane


# Polynomial value calculation
def Polyval(coeff, x):
    val = 0.0
    for i in range(np.size(coeff)):
        val = val + coeff[i] * x ** (np.size(coeff) - 1 - i)
    return val


# Global coordinate --> Local coordinate
def Global2Local(global_points, yaw_ego, X_ego, Y_ego):
    local_points = []
    num_data = len(global_points)
    theta = - yaw_ego
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for i in range(num_data):
        local_point = rotation_matrix @ (np.transpose([global_points[i][0] - X_ego, global_points[i][1] - Y_ego]))
        local_points.append(local_point.tolist())
    return local_points


# Polynomial fitting (n_th order)
def Polyfit(points, num_order):
    num_data = len(points)
    A_mat = np.zeros((num_data, num_order + 1))
    b = np.zeros((num_data, 1))
    # coeff = np.zeros((num_order+1, 1))
    for i in range(num_data):
        for j in range(num_order + 1):
            A_mat[i][j] = points[i][0] ** (num_order - j)
        b[i][0] = points[i][1]
    # coeff = np.linalg.inv(A_mat.T@A_mat)@A_mat.T@b
    coeff = np.linalg.pinv(A_mat) @ b
    return coeff.reshape(num_order + 1).tolist()


# Both lane to path
def BothLane2Path(coeff_L, coeff_R):
    coeff_path = []
    for i in range(len(coeff_L)):
        coeff_path.append((coeff_L[i] + coeff_R[i]) / 2)
    return coeff_path


# Vehicle model
class VehicleModel_Lat(object):
    def __init__(self, step_time, Vx, m=500, L=4, kv=0.005, Pos=[0.0, 0.0, 0.0]):
        self.dt = step_time
        self.m = m
        self.L = L
        self.kv = kv
        self.vx = Vx
        self.yawrate = 0
        self.Yaw = Pos[2]
        self.X = Pos[0]
        self.Y = Pos[1]

    def update(self, delta, Vx):
        self.vx = Vx
        self.delta = np.clip(delta, -0.5, 0.5)
        self.yawrate = self.vx / (self.L + self.kv * self.vx ** 2) * self.delta
        self.Yaw = self.Yaw + self.dt * self.yawrate
        self.X = self.X + Vx * self.dt * np.cos(self.Yaw)
        self.Y = self.Y + Vx * self.dt * np.sin(self.Yaw)


# Controller : Pure pursuit
class PurePursuit(object):
    def __init__(self, L=4.0, lookahead_time=1.0):
        self.L = L
        self.epsilon = 0.001
        self.t_lh = lookahead_time

    def ControllerInput(self, coeff, Vx):
        self.d_lh = Vx * self.t_lh
        self.y = Polyval(coeff, self.d_lh)
        self.u = np.arctan(2 * self.L * self.y / (self.d_lh ** 2 + self.y ** 2 + self.epsilon))


if __name__ == "__main__":
    step_time = 0.1
    simulation_time = 30.0
    Vx = 3.0
    X_lane = np.arange(0.0, 100.0, 0.1)
    Y_lane_L, Y_lane_R = lane(X_lane)

    ego_vehicle = VehicleModel_Lat(step_time, Vx)
    controller = PurePursuit()

    time = []
    X_ego = []
    Y_ego = []

    for i in range(int(simulation_time / step_time)):
        time.append(step_time * i)
        X_ego.append(ego_vehicle.X)
        Y_ego.append(ego_vehicle.Y)
        # Lane Info
        X_ref = np.arange(ego_vehicle.X, ego_vehicle.X + 5.0, 1.0)
        Y_ref_L, Y_ref_R = lane(X_ref)
        # Global points (front 5 meters from the ego vehicle)
        global_points_L = np.transpose(np.array([X_ref, Y_ref_L])).tolist()
        global_points_R = np.transpose(np.array([X_ref, Y_ref_R])).tolist()
        # Converted to local frame
        local_points_L = Global2Local(global_points_L, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        local_points_R = Global2Local(global_points_R, ego_vehicle.Yaw, ego_vehicle.X, ego_vehicle.Y)
        # 3th order fitting
        coeff_L = Polyfit(local_points_L, num_order=3)
        coeff_R = Polyfit(local_points_R, num_order=3)
        # Lane to path
        coeff_path = BothLane2Path(coeff_L, coeff_R)
        # Controller input
        controller.ControllerInput(coeff_path, Vx)
        ego_vehicle.update(controller.u, Vx)

    plt.figure(1, figsize=(13, 2))
    plt.plot(X_lane, Y_lane_L, 'k--')
    plt.plot(X_lane, Y_lane_R, 'k--', label="Reference")
    plt.plot(X_ego, Y_ego, 'b-', label="Vehicle Position")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="best")
    #    plt.axis("best")
    plt.grid(True)
    plt.show()

