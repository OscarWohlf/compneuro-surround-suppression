"""
Exercise 1 - Excitation-inhibition balance

Mini-project 1: Visual Surround Suppression using Excitation-Inhibition Circuits

This file solves:
- Ex. 1.1: sparse E/I connectivity matrix
- Ex. 1.2: f-I curves for E and I populations
- Ex. 1.4: effect of changing inhibition strength g
- Ex. 1.5: E-then-I stimulation experiment with raster and population rates

Run:
    python ex1_ei_balance.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix


# =========================
# Neuron parameters
# =========================

THETA = 20.0
U_RESET = -10.0
R = 1.0
DT = 0.5
TAU_M = 20.0

LAMBDA_BG = 25.0
BG_SCALE = 1.0

# =========================
# Network parameters
# =========================

NE = 1000
GAMMA = 0.25
NI = int(GAMMA * NE)

P = 0.02
KE = int(P * NE)
KI = int(P * NI)

J = 45.0
G_DEFAULT = 5.0

TAU_DELAY = 2.0
DELAY_STEPS = int(TAU_DELAY / DT)

FIGURE_DIR = "figures_ex1"


def make_output_folder():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def generate_sparse_connectivity(NE, NI, KE, KI, J, g, seed=0):
    """
    Generate sparse E/I connectivity matrix.

    Rows are postsynaptic neurons.
    Columns are presynaptic neurons.

    Excitatory neurons are indexed from 0 to NE - 1.
    Inhibitory neurons are indexed from NE to NE + NI - 1.

    Each postsynaptic neuron receives:
    - exactly KE excitatory incoming connections of strength J
    - exactly KI inhibitory incoming connections of strength -gJ
    """

    rng = np.random.default_rng(seed)

    N = NE + NI

    rows = []
    cols = []
    data = []

    excitatory_indices = np.arange(NE)
    inhibitory_indices = np.arange(NE, N)

    for post in range(N):
        presyn_E = rng.choice(excitatory_indices, size=KE, replace=False)
        presyn_I = rng.choice(inhibitory_indices, size=KI, replace=False)

        rows.extend([post] * KE)
        cols.extend(presyn_E)
        data.extend([J] * KE)

        rows.extend([post] * KI)
        cols.extend(presyn_I)
        data.extend([-g * J] * KI)

    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    return W


def simulate_ei_network(
    T,
    W,
    I_ext_E,
    I_ext_I,
    seed=0,
):
    """
    Simulate the full E/I network.

    Parameters
    ----------
    T : float
        Simulation time in ms.
    W : scipy sparse matrix
        Connectivity matrix of shape (N, N).
    I_ext_E : array of shape (n_steps,)
        External input to all E neurons.
    I_ext_I : array of shape (n_steps,)
        External input to all I neurons.
    seed : int
        Random seed.

    Returns
    -------
    times : array
    spikes : array of shape (n_steps, N)
        Boolean spike matrix.
    rate_E_inst : array
        Instantaneous excitatory population rate in Hz.
    rate_I_inst : array
        Instantaneous inhibitory population rate in Hz.
    """

    rng = np.random.default_rng(seed)

    N = NE + NI
    n_steps = int(T / DT)
    times = np.arange(n_steps) * DT

    u = rng.uniform(U_RESET, THETA, size=N)

    spikes = np.zeros((n_steps, N), dtype=bool)

    rate_E_inst = np.zeros(n_steps)
    rate_I_inst = np.zeros(n_steps)

    # Store previous spike trains for implementing synaptic delay
    delayed_spike_buffer = np.zeros((DELAY_STEPS + 1, N))

    for k in range(n_steps):
        delayed_index = k % (DELAY_STEPS + 1)
        delayed_spikes = delayed_spike_buffer[delayed_index]

        # Synaptic input:
        # W has units pC and spike train has 1/ms.
        # pC/ms = nA.
        I_syn = W @ delayed_spikes

        I_ext = np.zeros(N)
        I_ext[:NE] = I_ext_E[k]
        I_ext[NE:] = I_ext_I[k]

        I_bg = BG_SCALE * rng.poisson(LAMBDA_BG, size=N)

        I_total = I_syn + I_ext + I_bg

        du = (-u + R * I_total) / TAU_M
        u = u + DT * du

        spiked = u >= THETA
        spikes[k] = spiked

        # Store current spikes as spike train values, 1 / DT
        delayed_spike_buffer[delayed_index] = spiked.astype(float) / DT

        u[spiked] = U_RESET

        rate_E_inst[k] = np.sum(spiked[:NE]) / NE / (DT / 1000.0)
        rate_I_inst[k] = np.sum(spiked[NE:]) / NI / (DT / 1000.0)

    return times, spikes, rate_E_inst, rate_I_inst


def bin_population_rate(spikes, population_slice, bin_ms=20.0):
    """
    Compute binned population firing rate.

    Parameters
    ----------
    spikes : array of shape (n_steps, N)
        Boolean spike matrix.
    population_slice : slice
        Slice selecting E or I neurons.
    bin_ms : float
        Bin size in ms.

    Returns
    -------
    bin_centers : array
    binned_rates : array
        Rate in Hz.
    """

    bin_steps = int(bin_ms / DT)
    n_steps = spikes.shape[0]
    n_bins = n_steps // bin_steps

    selected = spikes[:, population_slice]
    n_neurons = selected.shape[1]

    rates = []
    centers = []

    for b in range(n_bins):
        start = b * bin_steps
        end = start + bin_steps

        spike_count = np.sum(selected[start:end])
        duration_s = bin_ms / 1000.0

        rate_hz = spike_count / n_neurons / duration_s

        center_ms = (start + end) / 2 * DT

        rates.append(rate_hz)
        centers.append(center_ms)

    return np.array(centers), np.array(rates)


def exercise_1_2():
    """
    Ex. 1.2:
    Plot f-I curves for excitatory and inhibitory populations.
    """

    print("Running Ex. 1.2...")

    W = generate_sparse_connectivity(
        NE=NE,
        NI=NI,
        KE=KE,
        KI=KI,
        J=J,
        g=G_DEFAULT,
        seed=10,
    )

    I_values = np.arange(-10.0, 50.0 + 0.1, 5.0)

    rates_E = []
    rates_I = []

    T = 100.0
    n_steps = int(T / DT)

    for I0 in I_values:
        I_ext_E = I0 * np.ones(n_steps)
        I_ext_I = I0 * np.ones(n_steps)

        times, spikes, rate_E_inst, rate_I_inst = simulate_ei_network(
            T=T,
            W=W,
            I_ext_E=I_ext_E,
            I_ext_I=I_ext_I,
            seed=int(I0 * 10 + 2000),
        )

        start_index = int(50.0 / DT)

        mean_E = np.mean(rate_E_inst[start_index:])
        mean_I = np.mean(rate_I_inst[start_index:])

        rates_E.append(mean_E)
        rates_I.append(mean_I)

    plt.figure(figsize=(8, 5))
    plt.plot(I_values, rates_E, "o-", label="Excitatory population")
    plt.plot(I_values, rates_I, "s-", label="Inhibitory population")
    plt.xlabel("Constant external current I0 (nA)")
    plt.ylabel("Mean firing rate over last 50 ms (Hz)")
    plt.title("Ex. 1.2 - f-I curves for E and I populations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex1_2_ei_fi_curves.png"), dpi=200)
    plt.close()

    print("Ex. 1.2 figure saved.")


def exercise_1_4():
    """
    Ex. 1.4:
    Test what happens when the inhibition-dominated condition is not satisfied.

    From the expected synaptic input:

        <I_syn> = J KE (r_E - g gamma r_I)

    If r_E and r_I are comparable, inhibition dominates when:

        g gamma > 1

    Since gamma = 0.25, this means g > 4.

    We compare g = 5 with smaller values.
    """

    print("Running Ex. 1.4...")

    g_values = [1.0, 2.0, 4.0, 5.0]
    T = 300.0
    n_steps = int(T / DT)

    I_ext_E = np.zeros(n_steps)
    I_ext_I = np.zeros(n_steps)

    mean_rates_E = []
    mean_rates_I = []

    for g in g_values:
        W = generate_sparse_connectivity(
            NE=NE,
            NI=NI,
            KE=KE,
            KI=KI,
            J=J,
            g=g,
            seed=20,
        )

        times, spikes, rate_E_inst, rate_I_inst = simulate_ei_network(
            T=T,
            W=W,
            I_ext_E=I_ext_E,
            I_ext_I=I_ext_I,
            seed=int(3000 + 10 * g),
        )

        start_index = int(150.0 / DT)

        mean_rates_E.append(np.mean(rate_E_inst[start_index:]))
        mean_rates_I.append(np.mean(rate_I_inst[start_index:]))

    x = np.arange(len(g_values))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, mean_rates_E, width, label="Excitatory")
    plt.bar(x + width / 2, mean_rates_I, width, label="Inhibitory")
    plt.axvline(2.5, linestyle="--", label="Boundary near g = 4")
    plt.xticks(x, [str(g) for g in g_values])
    plt.xlabel("Inhibition strength g")
    plt.ylabel("Mean firing rate after transient (Hz)")
    plt.title("Ex. 1.4 - Effect of changing inhibition strength")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex1_4_g_comparison.png"), dpi=200)
    plt.close()

    print("Ex. 1.4 figure saved.")


def exercise_1_5():
    """
    Ex. 1.5:
    First stimulate the excitatory population, then stimulate the inhibitory population.

    For 0 <= t < 500 ms:
        E receives I0(1 + sin(omega t))
        I receives 0

    For 500 <= t < 1000 ms:
        E receives 0
        I receives I0(1 + sin(omega t))
    """

    print("Running Ex. 1.5...")

    W = generate_sparse_connectivity(
        NE=NE,
        NI=NI,
        KE=KE,
        KI=KI,
        J=J,
        g=G_DEFAULT,
        seed=30,
    )

    T = 1000.0
    n_steps = int(T / DT)
    times = np.arange(n_steps) * DT
    times_s = times / 1000.0

    I0 = 10.0
    omega = 25.0

    oscillating_input = I0 * (1.0 + np.sin(omega * times_s))

    I_ext_E = np.zeros(n_steps)
    I_ext_I = np.zeros(n_steps)

    first_half = times < 500.0
    second_half = times >= 500.0

    I_ext_E[first_half] = oscillating_input[first_half]
    I_ext_I[second_half] = oscillating_input[second_half]

    times, spikes, rate_E_inst, rate_I_inst = simulate_ei_network(
        T=T,
        W=W,
        I_ext_E=I_ext_E,
        I_ext_I=I_ext_I,
        seed=40,
    )

    # Raster plot:
    # show 100 excitatory neurons and 25 inhibitory neurons
    E_to_show = np.arange(0, 100)
    I_to_show = np.arange(NE, NE + 25)

    plt.figure(figsize=(10, 5))

    E_spike_times, E_ids = np.where(spikes[:, E_to_show])
    I_spike_times, I_ids = np.where(spikes[:, I_to_show])

    plt.scatter(E_spike_times * DT, E_ids, s=5, label="E neurons")
    plt.scatter(I_spike_times * DT, I_ids + 100, s=5, label="I neurons")

    plt.axvline(500.0, linestyle="--", label="Switch stimulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Displayed neuron ID")
    plt.title("Ex. 1.5 - Raster plot, 100 E neurons and 25 I neurons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex1_5_raster.png"), dpi=200)
    plt.close()

    # Binned firing rates with 20 ms bins
    bin_centers_E, binned_E = bin_population_rate(spikes, slice(0, NE), bin_ms=20.0)
    bin_centers_I, binned_I = bin_population_rate(spikes, slice(NE, NE + NI), bin_ms=20.0)

    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers_E, binned_E, label="Excitatory population")
    plt.plot(bin_centers_I, binned_I, label="Inhibitory population")
    plt.axvline(500.0, linestyle="--", label="Switch stimulation")
    plt.xlabel("Time (ms)")
    plt.ylabel("Binned population firing rate (Hz)")
    plt.title("Ex. 1.5 - E/I firing rates with 20 ms bins")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex1_5_binned_rates.png"), dpi=200)
    plt.close()

    print("Ex. 1.5 figures saved.")


def main():
    make_output_folder()

    print("Network parameters:")
    print(f"NE = {NE}")
    print(f"NI = {NI}")
    print(f"KE = {KE}")
    print(f"KI = {KI}")
    print(f"J = {J}")
    print(f"g = {G_DEFAULT}")
    print(f"delay steps = {DELAY_STEPS}")

    exercise_1_2()
    exercise_1_4()
    exercise_1_5()

    print("All Ex. 1 simulations finished.")


if __name__ == "__main__":
    main()