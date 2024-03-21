import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np

class LowPassFilter:
    def __init__(self, y_initial_measure, alpha=0.9):
        # y_initial_measure를 numpy 배열로 변환
        self.y_estimate = np.array(y_initial_measure)
        self.alpha = alpha

    def estimate(self, y_measure):
        # y_measure를 numpy 배열로 변환
        y_measure = np.array(y_measure)
        # 이전 추정값과 측정값의 가중 평균을 계산하여 업데이트
        self.y_estimate = self.alpha * self.y_estimate + (1 - self.alpha) * y_measure


class KalmanFilter:
    def __init__(self, y_Measure_init, m=1.0, modelVariance=0.01, measureVariance=1.0, errorVariance_init=10.0):
        self.A = np.array([[1.0]])  # 상태 전이 행렬
        self.B = np.array([[1 / m]])  # 입력 제어 행렬
        self.C = np.array([[1.0]])  # 관측 행렬
        self.Q = np.array([[modelVariance]])  # 프로세스(모델) 잡음 공분산
        self.R = np.array([[measureVariance]])  # 측정 잡음 공분산
        self.x_estimate = np.array([[y_Measure_init]])  # 초기 상태 추정치
        self.P_estimate = np.array([[errorVariance_init]])  # 초기 오차 공분산

    def estimate(self, y_measure, input_u):
        # 입력과 측정값을 numpy 배열로 변환
        y_measure = np.array([[y_measure]])
        input_u = np.array([[input_u]])

        # Prediction
        self.x_prediction = self.A.dot(self.x_estimate) + self.B.dot(input_u)
        self.P_prediction = self.A.dot(self.P_estimate).dot(self.A.T) + self.Q

        # Update
        self.kalman_gain = self.P_prediction.dot(self.C.T).dot(
            np.linalg.inv(self.C.dot(self.P_prediction).dot(self.C.T) + self.R))
        self.x_estimate = self.x_prediction + self.kalman_gain.dot(y_measure - self.C.dot(self.x_prediction))
        self.P_estimate = (np.eye(self.A.shape[0]) - self.kalman_gain.dot(self.C)).dot(self.P_prediction)


class KalmanFilter2D:
    def __init__(self, initial_state, dt=1.0, modelVariance=0.01, measureVariance=1.0, errorVariance_init=1.0):
        # 상태 전이 행렬 A: 상태가 어떻게 시간에 따라 변하는지를 나타냅니다.
        # 여기서는 위치가 속도에 의해, 속도는 일정하다고 가정합니다(dt는 시간 간격).
        self.A = np.array([[1, dt],
                           [0, 1]])

        # 입력 제어 행렬 B: 이 예제에서는 외부에서 직접적인 제어가 없다고 가정합니다.
        self.B = np.array([[0],
                           [0]])

        # 관측 행렬 C: 실제 측정 가능한 변수와 상태 변수의 관계를 나타냅니다.
        # 여기서는 위치만 측정한다고 가정합니다.
        self.C = np.array([[1, 0]])

        # 프로세스(모델) 잡음 공분산 Q: 모델의 불확실성을 나타냅니다.
        self.Q = np.eye(2) * modelVariance

        # 측정 잡음 공분산 R: 측정 과정의 불확실성을 나타냅니다.
        self.R = np.array([[measureVariance]])

        # 초기 상태 추정치
        self.x_estimate = np.array(initial_state).reshape(2, 1)

        # 초기 오차 공분산
        self.P_estimate = np.eye(2) * errorVariance_init

    def estimate(self, y_measure):
        # 측정값을 numpy 배열로 변환 (여기서는 위치만 측정)
        y_measure = np.array([y_measure]).reshape(1, 1)

        # Prediction 단계
        self.x_prediction = np.dot(self.A, self.x_estimate)
        self.P_prediction = np.dot(np.dot(self.A, self.P_estimate), self.A.T) + self.Q

        # Update 단계
        S = np.dot(np.dot(self.C, self.P_prediction), self.C.T) + self.R
        self.kalman_gain = np.dot(np.dot(self.P_prediction, self.C.T), np.linalg.inv(S))
        self.x_estimate = self.x_prediction + np.dot(self.kalman_gain, (y_measure - np.dot(self.C, self.x_prediction)))
        self.P_estimate = np.dot((np.eye(2) - np.dot(self.kalman_gain, self.C)), self.P_prediction)

        return self.x_estimate.flatten()  # 상태 추정치 반환

class KalmanFilter4D:
    def __init__(self, initial_state, m=1.0, modelVariance=0.01, measureVariance=1.0, errorVariance_init=10.0):
        # 상태 전이 행렬 A: 시스템의 다음 상태가 현재 상태와 어떻게 연결되는지 나타냅니다.
        self.A = np.eye(4)  # 4x4 단위 행렬, 시간 스텝당 상태 변화가 크게 없다고 가정
        # 입력 제어 행렬 B: 외부 입력이 시스템 상태에 미치는 영향을 나타냅니다.
        self.B = np.zeros((4, 1))  # 여기서는 간단화를 위해 외부 입력의 직접적인 영향을 고려하지 않음
        # 관측 행렬 C: 실제 관측값이 상태 변수와 어떻게 연결되는지 나타냅니다.
        self.C = np.eye(4)  # 4x4 단위 행렬, 각 상태 변수가 직접 측정 가능하다고 가정
        # 프로세스(모델) 잡음 공분산 Q: 시스템 모델의 불확실성을 나타냅니다.
        self.Q = np.eye(4) * modelVariance
        # 측정 잡음 공분산 R: 측정 과정의 불확실성을 나타냅니다.
        self.R = np.eye(4) * measureVariance
        # 초기 상태 추정치
        self.x_estimate = np.array(initial_state).reshape(4, 1)
        # 초기 오차 공분산
        self.P_estimate = np.eye(4) * errorVariance_init

    def estimate(self, y_measure, input_u=0):
        # 측정값을 numpy 배열로 변환 (4x1 벡터로 가정)
        y_measure = np.array(y_measure).reshape(4, 1)
        # 외부 입력을 numpy 배열로 변환 (여기서는 입력을 고려하지 않으므로 0 벡터 사용)
        input_u = np.array([input_u]).reshape(1, 1)

        # Prediction 단계
        self.x_prediction = np.dot(self.A, self.x_estimate) + np.dot(self.B, input_u)
        self.P_prediction = np.dot(np.dot(self.A, self.P_estimate), self.A.T) + self.Q

        # Update 단계
        S = np.dot(np.dot(self.C, self.P_prediction), self.C.T) + self.R
        self.kalman_gain = np.dot(np.dot(self.P_prediction, self.C.T), np.linalg.inv(S))
        self.x_estimate = self.x_prediction + np.dot(self.kalman_gain, (y_measure - np.dot(self.C, self.x_prediction)))
        self.P_estimate = np.dot((np.eye(4) - np.dot(self.kalman_gain, self.C)), self.P_prediction)

        return self.x_estimate.flatten()  # 상태

# 사용 예시

# if __name__ == "__main__":
#     signal = pd.read_csv("01_Filter/Data/example_Filter_3.csv")
#
#     y_estimate = LowPassFilter(signal.y_measure[0])
#     for i, row in signal.iterrows():
#         y_estimate.estimate(signal.y_measure[i])
#         signal.y_estimate[i] = y_estimate.y_estimate
#
#     plt.figure()
#     plt.plot(signal.time, signal.y_measure, 'k.', label="Measure")
#     plt.plot(signal.time, signal.y_estimate, 'r-', label="Estimate")
#     plt.xlabel('time (s)')
#     plt.ylabel('signal')
#     plt.legend(loc="best")
#     plt.axis("equal")
#     plt.grid(True)
#     plt.show()



