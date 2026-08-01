import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter1D:

    def __init__(self, x0, P0, Q, R):
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R

    def predict(self):
        self.P = self.P + self.Q

    def update(self, measurement):

        x_pred = self.x
        P_pred = self.P

        K = P_pred / (P_pred + self.R)

        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return self.x

true_value = 50.0

np.random.seed(42)

num_steps = 50

measurements = true_value + np.random.normal(0, 5, num_steps)

kf = KalmanFilter1D(
    x0=0.0,
    P0=100.0,
    Q=0.1,
    R=25.0
)

estimates = []

for z in measurements:

    kf.predict()

    estimate = kf.update(z)

    estimates.append(estimate)

print(f"Final estimate : {kf.x:.4f}")
print(f"True value     : {true_value:.4f}")
print(f"Error          : {abs(kf.x-true_value):.4f}")

plt.figure(figsize=(10,5))

plt.plot(
    measurements,
    'x',
    alpha=0.5,
    label="Measurements"
)

plt.plot(
    estimates,
    linewidth=1.5,
    label="Kalman Estimate"
)

plt.axhline(
    true_value,
    color='green',
    linestyle='--',
    label="True Value"
)

plt.title("1D Kalman Filter")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
