import numpy as np

THETA = 20.0
U_RESET = -10.0
DT = 0.5
TAU_M = 20.0
R = 1.0
N_BG = 25.0

def init_neurons(N, u_reset=U_RESET, theta=THETA, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    neurons = rng.uniform(u_reset, theta, N)
    return neurons

def oscillating_input(I_0, omega, t_ms):
    t_sec = t_ms / 1000
    return I_0 * (1 + np.sin(omega * t_sec))

def delta_u(tau_m, curr_potentials, R, I):
    return (- curr_potentials + R * I) / tau_m

def simulate_lif_population(
    initial_u,
    t_max,
    external_current,
    dt=DT,
    theta=THETA,
    u_reset=U_RESET,
    tau_m=TAU_M,
    res=R,
    n_bg=None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    initial_u = np.asarray(initial_u, dtype=float)
    n_neurons = len(initial_u)
    n_steps = int(t_max / dt)
    times = np.arange(n_steps) * dt

    potentials = np.zeros((n_steps + 1, n_neurons))
    spikes = np.zeros((n_steps, n_neurons))

    potentials[0] = initial_u
    curr_u = initial_u.copy()

    external_current = np.asarray(external_current, dtype=float)

    if external_current.ndim == 0:
        external_current = np.full((n_steps, n_neurons), external_current)
    elif external_current.ndim == 1:
        external_current = external_current[:, None] * np.ones((1, n_neurons))

    for step in range(n_steps):
        current = external_current[step].copy()
        if n_bg is not None:
            current += rng.poisson(n_bg, size=n_neurons)

        d_u = delta_u(tau_m, curr_u, res, current)
        curr_u = curr_u + dt * d_u

        spiked = curr_u >= theta
        spikes[step, spiked] = 1 / dt
        curr_u[spiked] = u_reset
        potentials[step+ 1] = curr_u

    return times, potentials, spikes


def mean_rate_last_window(spikes, window_ms, dt=DT):
    n_neurons = spikes.shape[1]
    n_window_steps = int(window_ms / dt)

    spike_counts = np.sum(spikes[-n_window_steps:] > 0)
    duration_s = window_ms / 1000.0

    return spike_counts / n_neurons / duration_s

def population_rate_hz(spikes, dt=DT):
    n_neurons = spikes.shape[1]
    spike_counts = np.sum(spikes > 0, axis=1)
    return spike_counts / n_neurons / (dt / 1000.0)

def theoretical_lif_rate(i_values, theta=THETA, u_reset=U_RESET, tau_m=TAU_M, res=R):
    i_values = np.asarray(i_values, dtype=float)
    rates = np.zeros_like(i_values)
    for idx, I_0 in enumerate(i_values):
        if res * I_0 <= theta:
            rates[idx] = 0
        else:
            rates[idx] = 1000 / (tau_m * np.log((res * I_0 - u_reset) / (res * I_0 - (theta))))

    return rates
