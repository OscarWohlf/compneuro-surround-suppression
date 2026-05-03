"""
Exercise 2 - Field model of the cortical sheet

Mini-project 1: Visual Surround Suppression using Excitation-Inhibition Circuits

This file solves:
- Ex. 2.1: equations for unit-to-unit inputs are written in comments below
- Ex. 2.2: cortical sheet simulation made from multiple E-I units
- Ex. 2.3: unit-unit connectivity matrix on a ring
- Ex. 2.4: spontaneous bump simulation for W0 = 90 pC
- Ex. 2.5: sweep W0 to estimate the minimum value needed for bump formation

Run:
    python ex2_field_model.py

Important unit convention:
    The internal E-I unit returns firing rates in Hz for plotting, but the
    unit-to-unit synaptic current uses population spike activity in 1/ms.
    This is because W0 has units pC and pC/ms = nA.
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
# E-I unit parameters
# =========================

NE = 1000
GAMMA = 0.25
NI = int(GAMMA * NE)
N = NE + NI

P = 0.02
KE = int(P * NE)
KI = int(P * NI)

J = 45.0
G_DEFAULT = 5.0

TAU_DELAY = 2.0
DELAY_STEPS = int(TAU_DELAY / DT)

# =========================
# Cortical sheet parameters
# =========================

NUNITS = 10
SIGMA = 0.2
W0_DEFAULT = 90.0

FIGURE_DIR = "figures_ex2"


def make_output_folder():
    os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Reused sparse connectivity inside one E-I unit, from Ex. 1.1
# ============================================================


def generate_sparse_connectivity(NE, NI, KE, KI, J, g, seed=0):
    """
    Generate sparse E/I connectivity matrix for one E-I unit.

    Rows are postsynaptic neurons.
    Columns are presynaptic neurons.
    Excitatory neurons are indexed 0 ... NE-1.
    Inhibitory neurons are indexed NE ... NE+NI-1.
    """

    rng = np.random.default_rng(seed)

    total_n = NE + NI

    rows = []
    cols = []
    data = []

    excitatory_indices = np.arange(NE)
    inhibitory_indices = np.arange(NE, total_n)

    for post in range(total_n):
        presyn_E = rng.choice(excitatory_indices, size=KE, replace=False)
        presyn_I = rng.choice(inhibitory_indices, size=KI, replace=False)

        rows.extend([post] * KE)
        cols.extend(presyn_E)
        data.extend([J] * KE)

        rows.extend([post] * KI)
        cols.extend(presyn_I)
        data.extend([-g * J] * KI)

    return csr_matrix((data, (rows, cols)), shape=(total_n, total_n))


class EIUnit:
    """
    One E-I unit containing one excitatory and one inhibitory population.

    The method step(I_E, I_I) advances the local network by one time step.
    I_E is added to every excitatory neuron, and I_I is added to every
    inhibitory neuron. Both inputs are in nA.
    """

    def __init__(self, seed=0, g=G_DEFAULT):
        self.rng = np.random.default_rng(seed)
        self.W = generate_sparse_connectivity(
            NE=NE,
            NI=NI,
            KE=KE,
            KI=KI,
            J=J,
            g=g,
            seed=seed + 10000,
        )

        self.u = self.rng.uniform(U_RESET, THETA, size=N)

        # Exact delay of DELAY_STEPS time steps.
        # At step k, the buffer location k % DELAY_STEPS contains spikes
        # emitted DELAY_STEPS steps earlier.
        self.spike_buffer = np.zeros((DELAY_STEPS, N))
        self.step_count = 0

    def step(self, I_E, I_I):
        delayed_index = self.step_count % DELAY_STEPS
        delayed_spikes = self.spike_buffer[delayed_index]

        I_syn = self.W @ delayed_spikes

        I_ext = np.zeros(N)
        I_ext[:NE] = I_E
        I_ext[NE:] = I_I

        I_bg = BG_SCALE * self.rng.poisson(LAMBDA_BG, size=N)
        I_total = I_syn + I_ext + I_bg

        du = (-self.u + R * I_total) / TAU_M
        self.u = self.u + DT * du

        spiked = self.u >= THETA

        # Store current spikes as spike train values, 1 / ms.
        self.spike_buffer[delayed_index] = spiked.astype(float) / DT

        self.u[spiked] = U_RESET
        self.step_count += 1

        # Activity used by unit-unit coupling: mean spike train, units 1/ms.
        activity_E_per_ms = np.sum(spiked[:NE]) / NE / DT
        activity_I_per_ms = np.sum(spiked[NE:]) / NI / DT

        # Same activity converted to Hz for plotting/reporting.
        rate_E_hz = 1000.0 * activity_E_per_ms
        rate_I_hz = 1000.0 * activity_I_per_ms

        return activity_E_per_ms, activity_I_per_ms, rate_E_hz, rate_I_hz


# ============================================================
# Ex. 2.1 and 2.3: population-level field connectivity
# ============================================================


def ring_distance(x_alpha, x_beta):
    """Distance on a ring of circumference 1."""
    direct_distance = abs(x_alpha - x_beta)
    return min(direct_distance, 1.0 - direct_distance)


def generate_unit_connectivity(Nunits, sigma, W0, g, gamma):
    """
    Generate the 2*Nunits x 2*Nunits unit-unit connectivity matrix.

    Population ordering is:
        0 ... Nunits-1          : excitatory populations E_0 ... E_{N-1}
        Nunits ... 2*Nunits-1   : inhibitory populations I_0 ... I_{N-1}

    Rows are postsynaptic populations and columns are presynaptic populations.

    Equations implemented:
        W^{E<-E}_{alpha,beta} = W0 f(x_alpha, x_beta), alpha != beta
        W^{I<-E}_{alpha,beta} = g gamma W0 (1 - f(x_alpha, x_beta))
        W^{E<-I}_{alpha,beta} = W^{I<-I}_{alpha,beta} = 0

    where f = 1 if ring distance <= sigma, and 0 otherwise.
    """

    positions = np.linspace(0.0, 1.0, Nunits, endpoint=False)
    W_unit = np.zeros((2 * Nunits, 2 * Nunits))

    for alpha in range(Nunits):
        for beta in range(Nunits):
            d = ring_distance(positions[alpha], positions[beta])
            f = 1.0 if d <= sigma + 1e-12 else 0.0

            # E_alpha receives short-range excitation from E_beta.
            # Self-connections are excluded because local E-I dynamics already
            # contain within-unit interactions.
            if alpha != beta:
                W_unit[alpha, beta] = W0 * f

            # I_alpha receives excitation from far-away E_beta.
            # For beta = alpha, f = 1, so this is automatically zero.
            W_unit[Nunits + alpha, beta] = g * gamma * W0 * (1.0 - f)

            # Connections from inhibitory populations between units remain zero.
            W_unit[alpha, Nunits + beta] = 0.0
            W_unit[Nunits + alpha, Nunits + beta] = 0.0

    return W_unit, positions


# Ex. 2.1 equations, using the same population ordering as above:
#
# I_field,E^alpha(t) = sum_beta W_unit[alpha, beta] r_E^beta(t - tau_delay)
#                    + sum_beta W_unit[alpha, Nunits + beta] r_I^beta(t - tau_delay)
#
# I_field,I^alpha(t) = sum_beta W_unit[Nunits + alpha, beta] r_E^beta(t - tau_delay)
#                    + sum_beta W_unit[Nunits + alpha, Nunits + beta] r_I^beta(t - tau_delay)
#
# In the specific connectivity used here, the terms coming from inhibitory
# populations are zero, but the general equations are kept explicit.


# ============================================================
# Cortical sheet simulation
# ============================================================


def simulate_field_model(
    T,
    W_unit,
    seed=0,
    external_E=None,
    external_I=None,
):
    """
    Simulate a cortical sheet made of Nunits E-I units.

    Parameters
    ----------
    T : float
        Simulation time in ms.
    W_unit : array of shape (2*Nunits, 2*Nunits)
        Population-level unit-unit connectivity matrix, in pC.
    seed : int
        Random seed.
    external_E : None or array of shape (n_steps, Nunits)
        Additional external current to each E population, in nA.
    external_I : None or array of shape (n_steps, Nunits)
        Additional external current to each I population, in nA.

    Returns
    -------
    times : array of shape (n_steps,)
    rates_E_hz : array of shape (n_steps, Nunits)
    rates_I_hz : array of shape (n_steps, Nunits)
    field_inputs_E : array of shape (n_steps, Nunits)
    field_inputs_I : array of shape (n_steps, Nunits)
    """

    Nunits = W_unit.shape[0] // 2
    n_steps = int(T / DT)
    times = np.arange(n_steps) * DT

    if external_E is None:
        external_E = np.zeros((n_steps, Nunits))
    if external_I is None:
        external_I = np.zeros((n_steps, Nunits))

    units = [EIUnit(seed=seed + 100 * alpha) for alpha in range(Nunits)]

    rates_E_hz = np.zeros((n_steps, Nunits))
    rates_I_hz = np.zeros((n_steps, Nunits))
    field_inputs_E = np.zeros((n_steps, Nunits))
    field_inputs_I = np.zeros((n_steps, Nunits))

    # Delayed population activity buffer, units 1/ms.
    population_buffer = np.zeros((DELAY_STEPS, 2 * Nunits))

    for k in range(n_steps):
        delayed_index = k % DELAY_STEPS
        delayed_activity = population_buffer[delayed_index]

        # W_unit is pC and delayed_activity is 1/ms, so the product is nA.
        field_current = W_unit @ delayed_activity

        field_inputs_E[k] = field_current[:Nunits]
        field_inputs_I[k] = field_current[Nunits:]

        current_activity = np.zeros(2 * Nunits)

        for alpha, unit in enumerate(units):
            I_E = field_current[alpha] + external_E[k, alpha]
            I_I = field_current[Nunits + alpha] + external_I[k, alpha]

            a_E, a_I, r_E, r_I = unit.step(I_E=I_E, I_I=I_I)

            current_activity[alpha] = a_E
            current_activity[Nunits + alpha] = a_I
            rates_E_hz[k, alpha] = r_E
            rates_I_hz[k, alpha] = r_I

        population_buffer[delayed_index] = current_activity

    return times, rates_E_hz, rates_I_hz, field_inputs_E, field_inputs_I


def smooth_rates(rates, bin_ms=5.0):
    """
    Smooth population rates by non-overlapping time bins.
    This makes the 2D activity plots easier to read.
    """

    bin_steps = max(1, int(bin_ms / DT))
    n_steps, n_units = rates.shape
    n_bins = n_steps // bin_steps

    rates_trimmed = rates[: n_bins * bin_steps]
    rates_binned = rates_trimmed.reshape(n_bins, bin_steps, n_units).mean(axis=1)

    bin_centers = (np.arange(n_bins) * bin_steps + bin_steps / 2) * DT
    return bin_centers, rates_binned


def plot_activity_heatmap(times, rates_E_hz, filename, title):
    """Plot excitatory activity over time as a heatmap."""

    # Use 5 ms bins to reduce visual noise from instantaneous spikes.
    t_binned, rates_binned = smooth_rates(rates_E_hz, bin_ms=5.0)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        rates_binned.T,
        aspect="auto",
        origin="lower",
        extent=[t_binned[0], t_binned[-1], -0.5, rates_binned.shape[1] - 0.5],
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("E-I unit id")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Excitatory population activity (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=200)
    plt.close()


def detect_bump(rates_E_hz, analysis_start_ms=100.0):
    """
    Simple bump detector used for Ex. 2.5.

    A simulation is classified as having a bump if, after the transient,
    one unit has clearly larger activity than the sheet average and the
    activity profile is spatially localized.

    This is a qualitative detector; always inspect the heatmaps for the report.
    """

    start = int(analysis_start_ms / DT)
    mean_profile = rates_E_hz[start:].mean(axis=0)

    max_rate = np.max(mean_profile)
    mean_rate = np.mean(mean_profile)
    std_rate = np.std(mean_profile)

    # Localization score: coefficient of variation across units.
    cv = std_rate / (mean_rate + 1e-12)

    has_bump = (max_rate > 5.0) and (max_rate > 1.5 * mean_rate) and (cv > 0.25)
    bump_unit = int(np.argmax(mean_profile))

    return has_bump, bump_unit, max_rate, mean_rate, cv, mean_profile


# ============================================================
# Exercises
# ============================================================


def exercise_2_2_verify_no_interactions():
    """
    Ex. 2.2 verification: with W_unit = 0, units behave as independent
    balanced E-I networks driven only by background input.
    """

    print("Running Ex. 2.2 verification with unit-unit interactions disabled...")

    T = 200.0
    W_zero = np.zeros((2 * NUNITS, 2 * NUNITS))

    times, rates_E_hz, rates_I_hz, _, _ = simulate_field_model(
        T=T,
        W_unit=W_zero,
        seed=50,
    )

    start = int(100.0 / DT)
    mean_E_by_unit = rates_E_hz[start:].mean(axis=0)
    mean_I_by_unit = rates_I_hz[start:].mean(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(NUNITS), mean_E_by_unit, "o-", label="Excitatory")
    plt.plot(np.arange(NUNITS), mean_I_by_unit, "s-", label="Inhibitory")
    plt.xlabel("E-I unit id")
    plt.ylabel("Mean firing rate after transient (Hz)")
    plt.title("Ex. 2.2 - Check with unit-unit interactions disabled")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex2_2_no_unit_interactions.png"), dpi=200)
    plt.close()

    print("Mean E rates by unit, W_unit=0:", np.round(mean_E_by_unit, 2))
    print("Mean I rates by unit, W_unit=0:", np.round(mean_I_by_unit, 2))
    print("Ex. 2.2 verification figure saved.")


def exercise_2_3_plot_connectivity():
    """Plot the unit-unit connectivity matrix."""

    print("Running Ex. 2.3...")

    W_unit, positions = generate_unit_connectivity(
        Nunits=NUNITS,
        sigma=SIGMA,
        W0=W0_DEFAULT,
        g=G_DEFAULT,
        gamma=GAMMA,
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(W_unit, aspect="auto")
    plt.xlabel("Presynaptic population")
    plt.ylabel("Postsynaptic population")
    plt.title("Ex. 2.3 - Unit-unit connectivity matrix")
    cbar = plt.colorbar()
    cbar.set_label("Connection strength (pC)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex2_3_unit_connectivity.png"), dpi=200)
    plt.close()

    print("Ex. 2.3 connectivity figure saved.")
    return W_unit, positions


def exercise_2_4_spontaneous_bump():
    """Ex. 2.4: simulate spontaneous bump formation for W0 = 90 pC."""

    print("Running Ex. 2.4 spontaneous bump simulation...")

    W_unit, _ = generate_unit_connectivity(
        Nunits=NUNITS,
        sigma=SIGMA,
        W0=W0_DEFAULT,
        g=G_DEFAULT,
        gamma=GAMMA,
    )

    T = 200.0
    times, rates_E_hz, rates_I_hz, _, _ = simulate_field_model(
        T=T,
        W_unit=W_unit,
        seed=60,
    )

    plot_activity_heatmap(
        times,
        rates_E_hz,
        filename="ex2_4_spontaneous_bump_W0_90.png",
        title="Ex. 2.4 - Spontaneous bump, W0 = 90 pC",
    )

    has_bump, bump_unit, max_rate, mean_rate, cv, profile = detect_bump(rates_E_hz)

    # Estimate emergence time as the first 5-ms bin where the peak unit is
    # clearly above the sheet average.
    t_binned, rates_binned = smooth_rates(rates_E_hz, bin_ms=5.0)
    peak_by_time = rates_binned.max(axis=1)
    mean_by_time = rates_binned.mean(axis=1)
    emergence_candidates = np.where((peak_by_time > 5.0) & (peak_by_time > 1.5 * mean_by_time))[0]
    emergence_time = t_binned[emergence_candidates[0]] if len(emergence_candidates) > 0 else np.nan

    print(f"Bump detected: {has_bump}")
    print(f"Dominant bump unit: {bump_unit}")
    print(f"Estimated emergence time: {emergence_time:.1f} ms")
    print(f"Mean activity profile after 100 ms: {np.round(profile, 2)} Hz")
    print("Ex. 2.4 figure saved.")

    return rates_E_hz


def exercise_2_5_sweep_W0():
    """Ex. 2.5: repeat Ex. 2.4 for different W0 values."""

    print("Running Ex. 2.5 W0 sweep...")

    # A moderately fine grid. You can refine it around the transition once
    # you see where bump formation starts in your run.
    W0_values = np.arange(0.0, 121.0, 15.0)

    T = 200.0
    results = []

    for W0 in W0_values:
        W_unit, _ = generate_unit_connectivity(
            Nunits=NUNITS,
            sigma=SIGMA,
            W0=W0,
            g=G_DEFAULT,
            gamma=GAMMA,
        )

        times, rates_E_hz, _, _, _ = simulate_field_model(
            T=T,
            W_unit=W_unit,
            seed=100 + int(W0),
        )

        has_bump, bump_unit, max_rate, mean_rate, cv, profile = detect_bump(rates_E_hz)
        results.append((W0, has_bump, bump_unit, max_rate, mean_rate, cv))

        # Save a heatmap for every W0 so the threshold can be checked visually.
        plot_activity_heatmap(
            times,
            rates_E_hz,
            filename=f"ex2_5_activity_W0_{int(W0):03d}.png",
            title=f"Ex. 2.5 - Excitatory activity, W0 = {W0:.0f} pC",
        )

        print(
            f"W0={W0:5.1f} pC | bump={has_bump} | "
            f"unit={bump_unit} | max={max_rate:.2f} Hz | "
            f"mean={mean_rate:.2f} Hz | CV={cv:.2f}"
        )

    results = np.array(results, dtype=object)

    bump_W0 = [row[0] for row in results if row[1]]
    minimum_W0 = min(bump_W0) if len(bump_W0) > 0 else None

    # Summary plot for the report.
    W0_plot = np.array([row[0] for row in results], dtype=float)
    max_rates = np.array([row[3] for row in results], dtype=float)
    cvs = np.array([row[5] for row in results], dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(W0_plot, max_rates, "o-", label="Maximum unit activity")
    plt.xlabel("W0 (pC)")
    plt.ylabel("Maximum mean E activity after 100 ms (Hz)")
    plt.title("Ex. 2.5 - Bump strength as W0 is varied")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex2_5_W0_max_activity.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(W0_plot, cvs, "o-")
    plt.xlabel("W0 (pC)")
    plt.ylabel("Spatial coefficient of variation")
    plt.title("Ex. 2.5 - Spatial localization as W0 is varied")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex2_5_W0_localization.png"), dpi=200)
    plt.close()

    if minimum_W0 is None:
        print("No bump detected in this W0 range. Try extending the sweep above 120 pC.")
    else:
        print(f"Estimated minimum W0 for bump formation: {minimum_W0:.1f} pC")

    print("Ex. 2.5 figures saved.")
    return results, minimum_W0


def main():
    make_output_folder()

    print("Exercise 2 - Field model of the cortical sheet")
    print("Parameters:")
    print(f"NUNITS = {NUNITS}")
    print(f"NE = {NE}, NI = {NI}")
    print(f"sigma = {SIGMA}")
    print(f"W0 default = {W0_DEFAULT} pC")
    print(f"delay steps = {DELAY_STEPS}")

    exercise_2_2_verify_no_interactions()
    exercise_2_3_plot_connectivity()
    exercise_2_4_spontaneous_bump()
    exercise_2_5_sweep_W0()

    print("All Ex. 2 simulations finished.")


if __name__ == "__main__":
    main()
