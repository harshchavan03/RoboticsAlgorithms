import numpy as np

class KalmanFilter:
    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self):
        # Prediction step
        x_pred = self.x
        P_pred = self.P + self.Q

        self.x = x_pred
        self.P = P_pred

    def update(self, measurement):
        # Kalman Gain
        K = self.P / (self.P + self.R)

        # Update estimate
        self.x = self.x + K * (measurement - self.x)

        # Update covariance
        self.P = (1 - K) * self.P

    def get_state(self):
        return self.x

    def get_covariance(self):
        return self.P

true_value = 50.0

np.random.seed(42)
num_steps = 50

measurements = true_value + np.random.normal(0, 5, num_steps)

kf = KalmanFilter(
    x0=0.0,
    P0=100.0,
    Q=0.1,
    R=25.0
)

for z in measurements:
    kf.predict()
    kf.update(z)

final_estimate = kf.get_state()

print(f"Final estimate : {final_estimate:.4f}")
print(f"True value     : {true_value:.4f}")
print(f"Error          : {abs(final_estimate - true_value):.4f}")
