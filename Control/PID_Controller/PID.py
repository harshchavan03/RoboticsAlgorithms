import numpy as np
import matplotlib.pyplot as plt

setpoint = 10.0          # Desired target position
position = 0.0           # Initial position

Kp = 0.5                 # Proportional Gain

dt = 0.1                 # Time step
simulation_time = 10.0   # Seconds
num_steps = int(simulation_time / dt)

positions = []
errors = []
controls = []
time = []

for step in range(num_steps):

    current_time = step * dt

    # Compute Error
    error = setpoint - position

    # Proportional Controller
    control = Kp * error

    # Simple Plant Model
    # Position changes according to control signal
    position = position + control * dt

    # Store Results
    positions.append(position)
    errors.append(error)
    controls.append(control)
    time.append(current_time)

plt.figure(figsize=(10,5))
plt.plot(
    time,
    positions,
    linewidth=2,
    label="Plant Output"
)

plt.axhline(
    setpoint,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Setpoint"
)

plt.title("Proportional Controller Response")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.grid(True, alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig(
    "proportional_response.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.figure(figsize=(10,5))
plt.plot(
    time,
    errors,
    color="purple",
    linewidth=2,
    label="Tracking Error"
)

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.title("Tracking Error vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Error")
plt.grid(True, alpha=0.4)
plt.legend()

plt.tight_layout()
plt.savefig(
    "error_decay.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
