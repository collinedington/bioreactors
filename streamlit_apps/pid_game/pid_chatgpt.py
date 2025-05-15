import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Default PID constants and setpoint (percent)
DEFAULT_KP = 0.5  # updated default
DEFAULT_KI = 0.1  # updated default
DEFAULT_KD = 10.0  # updated default
DEFAULT_SETPOINT = 50.0  # %

# Initialize session state for parameters
for param, default in [('kp', DEFAULT_KP), ('ki', DEFAULT_KI), ('kd', DEFAULT_KD), ('setpoint', DEFAULT_SETPOINT)]:
    if param not in st.session_state:
        st.session_state[param] = default

# Reset callback
def reset_to_defaults():
    st.session_state['kp'] = DEFAULT_KP
    st.session_state['ki'] = DEFAULT_KI
    st.session_state['kd'] = DEFAULT_KD
    st.session_state['setpoint'] = DEFAULT_SETPOINT

# Sidebar controls
st.sidebar.title("PID Tuning Exercise")
st.sidebar.button("Reset to defaults", on_click=reset_to_defaults)
kp = st.sidebar.slider("Kp", 0.0, 5.0, st.session_state['kp'], key='kp')
ki = st.sidebar.slider("Ki", 0.0, 0.5, st.session_state['ki'], key='ki')
kd = st.sidebar.slider("Kd", 0.0, 25.0, st.session_state['kd'], key='kd')
setpoint_pct = st.sidebar.slider("Setpoint (%)", 0.0, 100.0, st.session_state['setpoint'], key='setpoint')

st.title("Interactive PID Tuning Example")

# Simulation parameters
total_time = 150.0  # total time units
n_points = 1000

time = np.linspace(0, total_time, n_points)
dt = time[1] - time[0]

# Second-order underdamped plant parameters
wn = 0.1        # reduced natural frequency for more lag
zeta = 0.2      # damping ratio (zeta < 1 for oscillation potential)
b = 1.0         # plant gain

# Arrays for states and control
y = np.zeros_like(time)    # DO level (fraction)
y_dot = np.zeros_like(time)
u = np.zeros_like(time)  # control action

# PID controller state
integral = 0.0
prev_error = 0.0

# Cell oxygen consumption parameters
consumption_rate = 0.01        # mean consumption per time unit (fraction)
consumption_noise_std = 0.002  # std dev of consumption noise

# Convert setpoint to fraction
setpoint = setpoint_pct / 100.0

# Simulation loop
for i in range(1, len(time)):
    # noisy consumption
    consumption = consumption_rate + np.random.normal(0, consumption_noise_std)

    # PID control calculations
    error = setpoint - y[i-1]
    integral += error * dt
    derivative = (error - prev_error) / dt
    prev_error = error

    # compute control action
    u[i] = kp * error + ki * integral + kd * derivative
    u[i] = max(u[i], 0.0)

    # plant state updates (second-order)
    dy = y_dot[i-1]
    dy_dot = -2 * zeta * wn * y_dot[i-1] - wn**2 * y[i-1] + wn**2 * b * u[i]

    # integrate states and subtract consumption
    y[i] = y[i-1] + dy * dt - consumption * dt
    y_dot[i] = y_dot[i-1] + dy_dot * dt

    # clamp between 0 and 1
    y[i] = np.clip(y[i], 0.0, 1.0)

# Convert to percentage
y_pct = y * 100.0

# Plot DO response
fig, ax = plt.subplots()
ax.plot(time, y_pct, label="Output Level")
ax.axhline(setpoint_pct, color='r', linestyle='--', label="Setpoint")
ax.set_xlabel("Time")
ax.set_ylabel("Response (%)")
ax.set_ylim(0, 100)
ax.set_title(f"Response with Kp={kp}, Ki={ki}, Kd={kd}, Setpoint={setpoint_pct}%")
ax.legend()

# Display plot
st.pyplot(fig)
