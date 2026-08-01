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

configs = [

    ("Default\nR=25, Q=0.1", 0.1, 25.0, "red"),

    ("High R=400\nTrust Model more", 0.1, 400.0, "orange"),

    ("Low R=1\nTrust Sensor more", 0.1, 1.0, "purple"),

    ("High Q=10\n Model drifts/responds quickly", 10.0, 25.0, "blue")

]

fig, axes = plt.subplots(2, 2, figsize=(13,8))

for ax, (title, Q, R, color) in zip(axes.flatten(), configs):

    kf = KalmanFilter1D(
        x0=0.0,
        P0=100.0,
        Q=Q,
        R=R
    )

    estimates = []

    for z in measurements:

        kf.predict()

        estimate = kf.update(z)

        estimates.append(estimate)

    ax.plot(
        measurements,
        'x',
        color='gray',
        markersize=4,
        alpha=0.5,
        label="Noisy Measurements"
    )

    ax.plot(
        estimates,
        color=color,
        linewidth=2.0,
        label="KF Estimate"
    )

    ax.axhline(
        true_value,
        color='green',
        linestyle='--',
        linewidth=1.5,
        label="True Value"
    )

    ax.set_title(
        title,
        fontsize=10
    )

    ax.set_xlabel("Timestep")

    ax.set_ylabel("Value")

    ax.grid(
        True,
        alpha=0.5
    )

    ax.legend(
        fontsize=8
    )


plt.suptitle(
    "1D KF— Parameter Sensitivity(4 Graph Visualization)",
    fontsize=15,
)
plt.tight_layout()
plt.show()
