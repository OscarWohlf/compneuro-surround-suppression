import numpy as np
from src.parameters import (
    THETA,
    U_RESET,
    DT,
    TAU_M,
    R,
    N_BG,
    N_E,
    N_I,
    N,
    TAU_DELAY,
)
from src.lif import init_neurons, delta_u

def synaptic_input(step, w, spikes, tau_delay=TAU_DELAY, dt = DT):
    I_syn = np.zeros(w.shape[0])
    delay_steps = int(tau_delay / dt)
    delayed_step = step - delay_steps
    if delayed_step < 0:
        return np.zeros(w.shape[0])
    
    I_syn = w @ spikes[delayed_step]
    return I_syn


def simulate_ei_network(t_max, w, i_ext_e, i_ext_i, initial_u=None, n_e=N_E,n_i=N_I, 
    dt=DT, theta=THETA, u_reset=U_RESET, tau_m=TAU_M, res=R, n_bg=N_BG, tau_delay=TAU_DELAY, rng=None,):
    if rng is None:
        rng = np.random.default_rng()

    n_total = n_e + n_i
    n_steps = int(t_max / dt)
    times = np.arange(n_steps) * dt

    i_ext_e = _prepare_population_input(i_ext_e, n_steps)
    i_ext_i = _prepare_population_input(i_ext_i, n_steps)

    if initial_u is None:
        curr_u = init_neurons(n_total, rng=rng)
    else:
        curr_u = np.asarray(initial_u, dtype=float).copy()

    potentials = np.zeros((n_steps + 1, n_total))
    spikes = np.zeros((n_steps, n_total))
    potentials[0] = curr_u

    rate_e = np.zeros(n_steps)
    rate_i = np.zeros(n_steps)

    for step in range(n_steps):
        i_syn = synaptic_input(step, w, spikes, tau_delay=tau_delay, dt=dt)

        i_ext = np.zeros(n_total)
        i_ext[:n_e] = i_ext_e[step]
        i_ext[n_e:] = i_ext_i[step]

        i_bg = rng.poisson(n_bg, size=n_total)

        i_total = i_syn + i_ext + i_bg

        du = delta_u(tau_m, curr_u, res, i_total)
        curr_u = curr_u + dt * du

        spiked = curr_u >= theta
        spikes[step, spiked] = 1.0 / dt
        curr_u[spiked] = u_reset

        potentials[step + 1] = curr_u

        rate_e[step] = np.sum(spiked[:n_e]) / n_e / (dt / 1000.0)
        rate_i[step] = np.sum(spiked[n_e:]) / n_i / (dt / 1000.0)
    
    return times, potentials, spikes, rate_e, rate_i


def bin_population_rate(spikes, population_slice, bin_ms=20.0, dt=DT):
    bin_steps = int(bin_ms / dt)
    n_steps = spikes.shape[0]
    n_bins = n_steps // bin_steps

    selected_spikes = spikes[:, population_slice] > 0
    selected_spikes = selected_spikes[: n_bins * bin_steps]

    n_neurons = selected_spikes.shape[1]

    reshaped = selected_spikes.reshape(n_bins, bin_steps, n_neurons)
    spike_counts = reshaped.sum(axis=(1, 2))

    duration_s = bin_ms / 1000.0
    rates = spike_counts / n_neurons / duration_s

    bin_centers = np.arange(n_bins) * bin_ms + bin_ms / 2

    return bin_centers, rates


def _prepare_population_input(input_current, n_steps):
    input_current = np.asarray(input_current, dtype=float)

    if input_current.ndim == 0:
        return np.full(n_steps, input_current)

    if input_current.shape[0] != n_steps:
        raise ValueError("Input current must have one value per time step.")

    return input_current