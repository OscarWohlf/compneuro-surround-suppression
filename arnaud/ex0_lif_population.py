"""
Exercise 0 - Population of unconnected LIF neurons

Mini-project 1: Visual Surround Suppression using Excitation-Inhibition Circuits

This file solves:
- Ex. 0.1: raster plot and mean firing rate for 100 unconnected LIF neurons
- Ex. 0.2: f-I curve with stochastic background input and comparison with theory

Run:
    python ex0_lif_population.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Global neuron parameters
# =========================

THETA = 20.0          # threshold, mV
U_RESET = -10.0       # reset potential, mV
R = 1.0               # membrane resistance, MOhm
DT = 0.5              # time step, ms
TAU_M = 20.0          # membrane time constant, ms

LAMBDA_BG = 25.0      # Poisson background rate parameter
BG_SCALE = 1.0        # 1 nA * Poisson(lambda_bg)

FIGURE_DIR = "figures_ex0"


def make_output_folder():
    """Create the folder where figures will be saved."""
    os.makedirs(FIGURE_DIR, exist_ok=True)


def simulate_lif_population(
    initial_u,
    T,
    external_current,
    use_background=False,
    seed=0,
):
    """
    Simulate a population of unconnected LIF neurons.

    Parameters
    ----------
    initial_u : array of shape (N,)
        Initial membrane potentials.
    T : float
        Total simulation time in ms.
    external_current : array of shape (n_steps,) or (n_steps, N)
        External input current in nA.
    use_background : bool
        If True, adds stochastic Poisson background input.
    seed : int
        Random seed.

    Returns
    -------
    times : array of shape (n_steps,)
        Time values in ms.
    potentials : array of shape (n_steps, N)
        Membrane potentials over time.
    spikes : array of shape (n_steps, N)
        Spike train. A spike is represented as 1 / DT, as requested.
    """

    rng = np.random.default_rng(seed)

    initial_u = np.asarray(initial_u, dtype=float)
    N = len(initial_u)

    n_steps = int(T / DT)
    times = np.arange(n_steps) * DT

    potentials = np.zeros((n_steps, N))
    spikes = np.zeros((n_steps, N))

    u = initial_u.copy()

    # Make sure external_current has shape (n_steps, N)
    external_current = np.asarray(external_current)

    if external_current.ndim == 1:
        external_current = external_current[:, None] * np.ones((1, N))

    for k in range(n_steps):
        potentials[k] = u

        I_total = external_current[k].copy()

        if use_background:
            I_bg = BG_SCALE * rng.poisson(LAMBDA_BG, size=N)
            I_total += I_bg

        # Forward Euler update:
        # tau_m du/dt = -u + R I
        du = (-u + R * I_total) / TAU_M
        u = u + DT * du

        # Detect threshold crossing
        spiked = u >= THETA

        # Spike train value is 1 / DT when the neuron spikes
        spikes[k, spiked] = 1.0 / DT

        # Reset after spike
        u[spiked] = U_RESET

    return times, potentials, spikes


def theoretical_lif_rate(I_values):
    """
    Theoretical f-I curve for a deterministic LIF neuron without background input.

    For constant input I, if R*I <= theta, the neuron never reaches threshold.
    Otherwise:

        T_spike = tau_m * ln((R I - u_reset) / (R I - theta))

    Rate is returned in Hz.
    """

    rates = np.zeros_like(I_values, dtype=float)

    for idx, I in enumerate(I_values):
        drive = R * I

        if drive <= THETA:
            rates[idx] = 0.0
        else:
            isi_ms = TAU_M * np.log((drive - U_RESET) / (drive - THETA))
            rates[idx] = 1000.0 / isi_ms

    return rates


def exercise_0_1():
    """
    Ex. 0.1:
    Simulate N = 100 unconnected LIF neurons receiving a slowly oscillating input.
    """

    print("Running Ex. 0.1...")

    N = 100
    T = 1000.0

    I0 = 20.0              # nA
    omega = 10.0           # rad/s

    rng = np.random.default_rng(1)
    initial_u = rng.uniform(U_RESET, THETA, size=N)

    n_steps = int(T / DT)
    times = np.arange(n_steps) * DT

    # Convert ms to seconds for omega*t
    times_s = times / 1000.0

    I_ext = I0 * (1.0 + np.sin(omega * times_s))

    times, potentials, spikes = simulate_lif_population(
        initial_u=initial_u,
        T=T,
        external_current=I_ext,
        use_background=False,
        seed=2,
    )

    # Raster plot
    spike_times, neuron_ids = np.where(spikes > 0)
    spike_times_ms = spike_times * DT

    plt.figure(figsize=(10, 5))
    plt.scatter(spike_times_ms, neuron_ids, s=6)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.title("Ex. 0.1 - Raster plot of 100 unconnected LIF neurons")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex0_1_raster.png"), dpi=200)
    plt.close()

    # Mean firing rate over time
    # spikes has value 1/DT, but for population rate per neuron in Hz:
    # count spikes in each time bin / N / bin duration in seconds
    spike_counts_per_step = np.sum(spikes > 0, axis=1)
    population_rate_hz = spike_counts_per_step / N / (DT / 1000.0)

    # Normalize input only for visual comparison
    I_ext_normalized = I_ext / np.max(I_ext) * np.max(population_rate_hz)

    plt.figure(figsize=(10, 5))
    plt.plot(times, population_rate_hz, label="Mean firing rate")
    plt.plot(times, I_ext_normalized, "--", label="External input, rescaled")
    plt.xlabel("Time (ms)")
    plt.ylabel("Population firing rate (Hz)")
    plt.title("Ex. 0.1 - Mean firing rate compared with input")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex0_1_rate_vs_input.png"), dpi=200)
    plt.close()

    print("Ex. 0.1 figures saved.")


def exercise_0_2():
    """
    Ex. 0.2:
    Plot the f-I curve for N = 100 neurons with stochastic background input.
    """

    print("Running Ex. 0.2...")

    N = 100
    T = 100.0

    I_values = np.arange(-10.0, 50.0 + 0.1, 5.0)
    simulated_rates = []

    rng = np.random.default_rng(3)

    for I0 in I_values:
        initial_u = rng.uniform(U_RESET, THETA, size=N)

        n_steps = int(T / DT)
        I_ext = I0 * np.ones(n_steps)

        times, potentials, spikes = simulate_lif_population(
            initial_u=initial_u,
            T=T,
            external_current=I_ext,
            use_background=True,
            seed=int(I0 * 10 + 1000),
        )

        # Use the last 50 ms only
        start_index = int(50.0 / DT)
        spike_counts = np.sum(spikes[start_index:] > 0)

        duration_s = 50.0 / 1000.0
        mean_rate_hz = spike_counts / N / duration_s

        simulated_rates.append(mean_rate_hz)

    simulated_rates = np.array(simulated_rates)
    theory_rates = theoretical_lif_rate(I_values)

    plt.figure(figsize=(8, 5))
    plt.plot(I_values, simulated_rates, "o-", label="Simulation with background input")
    plt.plot(I_values, theory_rates, "s--", label="Theory without background input")
    plt.xlabel("Constant external current I0 (nA)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title("Ex. 0.2 - f-I curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex0_2_fi_curve.png"), dpi=200)
    plt.close()

    print("Ex. 0.2 figure saved.")


def main():
    make_output_folder()
    exercise_0_1()
    exercise_0_2()
    print("All Ex. 0 simulations finished.")


if __name__ == "__main__":
    main()