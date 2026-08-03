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

        return self.x, K


true_value = 50.0
np.random.seed(42)
num_steps = 50
measurements = true_value + np.random.normal(0, 5, num_steps)

configs = [

    ("Default\nR=25, Q=0.1", 0.1, 25.0, "red"),

    ("High R=400\nTrust Model more", 0.1, 400.0, "orange"),

    ("Low R=1\nTrust Sensor more", 0.1, 1.0, "purple"),

    ("High Q=10\nModel drifts/responds quickly", 10.0, 25.0, "blue")

]

fig, axes = plt.subplots(2, 2, figsize=(13,8))
default_gains = []
default_estimates = []

for ax, (title, Q, R, color) in zip(axes.flatten(), configs):

    kf = KalmanFilter1D(
        x0=0.0,
        P0=100.0,
        Q=Q,
        R=R
    )

    estimates = []
    gains = []

    for z in measurements:

        kf.predict()

        estimate, gain = kf.update(z)

        estimates.append(estimate)
        gains.append(gain)

    if Q == 0.1 and R == 25.0:
        default_gains = gains.copy()
        default_estimates = estimates.copy()

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
    "1D KF — Parameter Sensitivity (4 Graph Visualization)",
    fontsize=15,
)

plt.tight_layout()
plt.savefig(
    "parameter_sensitivity_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.figure(figsize=(10,5))

plt.plot(
    default_gains,
    color="royalblue",
    linewidth=2.5,
    label="Kalman Gain"
)

plt.title(
    "Kalman Gain Convergence (Default: Q=0.1, R=25)"
)

plt.xlabel("Timestep")
plt.ylabel("Kalman Gain (K)")
plt.grid(
    True,
    alpha=0.5
)
plt.legend()
plt.tight_layout()
plt.savefig(
    "kalman_gain_convergence.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

measurement_rmse = np.sqrt(
    np.mean((measurements - true_value) ** 2)
)

estimate_rmse = np.sqrt(
    np.mean((np.array(default_estimates) - true_value) ** 2)
)

improvement = (
    (measurement_rmse - estimate_rmse)
    / measurement_rmse
) * 100

print("\nRMSE PERFORMANCE")

print(f"Measurement RMSE : {measurement_rmse:.3f}")

print(f"KF Estimate RMSE : {estimate_rmse:.3f}")

print(f"Improvement      : {improvement:.2f}%")

plt.figure(figsize=(7,5))

bars = plt.bar(

    ["Measurements", "Kalman Estimate"],

    [measurement_rmse, estimate_rmse],

    color=["gray", "royalblue"]

)

plt.ylabel("RMSE")
plt.title("RMSE Comparison")
plt.grid(
    axis="y",
    alpha=0.4
)

for bar in bars:

    height = bar.get_height()
    plt.text(

        bar.get_x() + bar.get_width()/2,

        height + 0.05,

        f"{height:.2f}",

        ha="center",

        fontsize=10

    )

plt.tight_layout()
plt.savefig(

    "rmse_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()
