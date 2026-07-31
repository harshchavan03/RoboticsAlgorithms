import numpy as np

# --- True value ---
true_value = 50.0

# --- Simulated noisy measurements ---
np.random.seed(42)
num_steps = 50
measurements = true_value + np.random.normal(0, 5, num_steps)

# --- Filter parameters ---
x = 0.0      # initial estimate
P = 100.0    # initial uncertainty
Q = 0.1      # process noise
R = 25.0     # measurement noise variance (std=5, var=25)

# --- Filter loop ---
estimates = []

for z in measurements:
    # PREDICT
    x_pred = x
    P_pred = P + Q

    # UPDATE
    K = P_pred / (P_pred + R)
    x = x_pred + K * (z - x_pred)
    P = (1 - K) * P_pred

    estimates.append(x)

print(f"Final estimate : {x:.4f}")
print(f"True value     : {true_value:.4f}")
print(f"Error          : {abs(x - true_value):.4f}")
