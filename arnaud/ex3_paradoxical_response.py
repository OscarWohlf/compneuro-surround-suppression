"""
Exercise 3 - Paradoxical response properties

Mini-project 1: Visual Surround Suppression using Excitation-Inhibition Circuits

This file solves:
- Ex. 3.1: step stimulation of one E population with W0 below bump threshold
- Ex. 3.2: spatial response profile / tuning curve interpretation
- Ex. 3.3: response of one E population as stimulus width changes
- Ex. 3.4 and Ex. 3.5: diagnostic plots and printed interpretation

Run:
    python ex3_paradoxical_response.py

Important convention:
    The unit-unit connectivity W_unit has units pC. Therefore it must multiply
    population spike activity in 1/ms, not firing rates in Hz, because pC/ms = nA.
    The code stores rates in Hz only for plotting and reporting.
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
SIGMA = 0.2          # range of short-range E-to-E unit interactions
W0_EX3 = 45.0        # below the spontaneous bump threshold estimated in Ex. 2
I0_STIM = 30.0       # nA
TARGET_UNIT = 5

FIGURE_DIR = "figures_ex3"


def make_output_folder():
    os.makedirs(FIGURE_DIR, exist_ok=True)


# ============================================================
# Local sparse E-I unit, same idea as Ex. 1 and Ex. 2
# ============================================================


def generate_sparse_connectivity(NE, NI, KE, KI, J, g, seed=0):
    """
    Generate sparse E/I connectivity matrix for one local E-I unit.

    Rows are postsynaptic neurons. Columns are presynaptic neurons.
    Excitatory neurons are 0 ... NE-1, inhibitory neurons are NE ... NE+NI-1.
    Each neuron receives exactly KE excitatory and KI inhibitory inputs.
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
    One local E-I unit containing one excitatory and one inhibitory population.

    step(I_E, I_I) advances this local network by one time step.
    I_E is added to every E neuron, and I_I to every I neuron.
    Both inputs are in nA.
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

        # Store current spikes as spike train values in 1/ms.
        self.spike_buffer[delayed_index] = spiked.astype(float) / DT
        self.u[spiked] = U_RESET
        self.step_count += 1

        # Population activity used by unit-unit coupling: mean spike train in 1/ms.
        activity_E_per_ms = np.sum(spiked[:NE]) / NE / DT
        activity_I_per_ms = np.sum(spiked[NE:]) / NI / DT

        # Same activity in Hz for plotting.
        rate_E_hz = 1000.0 * activity_E_per_ms
        rate_I_hz = 1000.0 * activity_I_per_ms

        return activity_E_per_ms, activity_I_per_ms, rate_E_hz, rate_I_hz


# ============================================================
# Cortical sheet connectivity
# ============================================================


def ring_distance(x_alpha, x_beta):
    """Distance on a ring of circumference 1."""
    direct_distance = abs(x_alpha - x_beta)
    return min(direct_distance, 1.0 - direct_distance)


def generate_unit_connectivity(Nunits, sigma, W0, g, gamma):
    """
    Generate the 2*Nunits x 2*Nunits population connectivity matrix.

    Population ordering:
        0 ... Nunits-1          : E_0 ... E_{Nunits-1}
        Nunits ... 2*Nunits-1   : I_0 ... I_{Nunits-1}

    Rows are postsynaptic populations and columns are presynaptic populations.

    Implemented connections:
        W^{E<-E}_{alpha,beta} = W0 f(x_alpha, x_beta), alpha != beta
        W^{I<-E}_{alpha,beta} = g gamma W0 (1 - f(x_alpha, x_beta))
        W^{E<-I}_{alpha,beta} = W^{I<-I}_{alpha,beta} = 0

    f = 1 when ring distance <= sigma, otherwise f = 0.
    """

    positions = np.linspace(0.0, 1.0, Nunits, endpoint=False)
    W_unit = np.zeros((2 * Nunits, 2 * Nunits))

    for alpha in range(Nunits):
        for beta in range(Nunits):
            d = ring_distance(positions[alpha], positions[beta])
            f = 1.0 if d <= sigma + 1e-12 else 0.0

            # Short-range excitation between E populations.
            # Exclude self-connections because each E-I unit already has local recurrent connectivity.
            if alpha != beta:
                W_unit[alpha, beta] = W0 * f

            # Long-range inhibitory effect is implemented indirectly:
            # far-away E_beta excites I_alpha, which then locally inhibits E_alpha.
            W_unit[Nunits + alpha, beta] = g * gamma * W0 * (1.0 - f)

    return W_unit, positions


# ============================================================
# Simulation and plotting helpers
# ============================================================


def simulate_field_model(T, W_unit, seed=0, external_E=None, external_I=None):
    """
    Simulate a cortical sheet made from multiple E-I units.

    external_E and external_I have shape (n_steps, Nunits) and units nA.
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
    total_inputs_E = np.zeros((n_steps, Nunits))
    total_inputs_I = np.zeros((n_steps, Nunits))

    # Delayed population activity in 1/ms.
    population_buffer = np.zeros((DELAY_STEPS, 2 * Nunits))

    for k in range(n_steps):
        delayed_index = k % DELAY_STEPS
        delayed_activity = population_buffer[delayed_index]

        # W_unit is pC and delayed_activity is 1/ms, so field_current is nA.
        field_current = W_unit @ delayed_activity
        field_inputs_E[k] = field_current[:Nunits]
        field_inputs_I[k] = field_current[Nunits:]

        current_activity = np.zeros(2 * Nunits)

        for alpha, unit in enumerate(units):
            I_E = field_current[alpha] + external_E[k, alpha]
            I_I = field_current[Nunits + alpha] + external_I[k, alpha]

            total_inputs_E[k, alpha] = I_E
            total_inputs_I[k, alpha] = I_I

            a_E, a_I, r_E, r_I = unit.step(I_E=I_E, I_I=I_I)

            current_activity[alpha] = a_E
            current_activity[Nunits + alpha] = a_I
            rates_E_hz[k, alpha] = r_E
            rates_I_hz[k, alpha] = r_I

        population_buffer[delayed_index] = current_activity

    return {
        "times": times,
        "rates_E_hz": rates_E_hz,
        "rates_I_hz": rates_I_hz,
        "field_inputs_E": field_inputs_E,
        "field_inputs_I": field_inputs_I,
        "total_inputs_E": total_inputs_E,
        "total_inputs_I": total_inputs_I,
    }


def smooth_rates(rates, bin_ms=5.0):
    """Average rates in non-overlapping bins for clearer plots."""

    bin_steps = max(1, int(bin_ms / DT))
    n_steps, n_units = rates.shape
    n_bins = n_steps // bin_steps

    rates_trimmed = rates[: n_bins * bin_steps]
    rates_binned = rates_trimmed.reshape(n_bins, bin_steps, n_units).mean(axis=1)
    bin_centers = (np.arange(n_bins) * bin_steps + bin_steps / 2) * DT

    return bin_centers, rates_binned


def plot_activity_heatmap(times, rates_E_hz, filename, title, stimulus_unit=None, stimulus_onset=None):
    """Plot excitatory population activity as a time x unit heatmap."""

    t_binned, rates_binned = smooth_rates(rates_E_hz, bin_ms=5.0)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        rates_binned.T,
        aspect="auto",
        origin="lower",
        extent=[t_binned[0], t_binned[-1], -0.5, rates_binned.shape[1] - 0.5],
    )
    if stimulus_onset is not None:
        plt.axvline(stimulus_onset, linestyle="--", linewidth=1.5, label="Stimulus onset")
    if stimulus_unit is not None:
        plt.axhline(stimulus_unit, linestyle=":", linewidth=1.5, label="Stimulated unit")
        plt.legend(loc="upper right")
    plt.xlabel("Time (ms)")
    plt.ylabel("E-I unit id")
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label("Excitatory population activity (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=200)
    plt.close()


def gaussian_stimulus_profile(positions, center_position, sigma_stim, I0):
    """
    Gaussian stimulus on a ring.

    For sigma_stim = 0, stimulate only the closest unit to the center.
    """

    distances = np.array([ring_distance(x, center_position) for x in positions])

    if sigma_stim == 0:
        profile = np.zeros_like(positions)
        profile[np.argmin(distances)] = I0
        return profile

    return I0 * np.exp(-(distances ** 2) / (2.0 * sigma_stim ** 2))


# ============================================================
# Exercise 3
# ============================================================


def exercise_3_1_and_3_2():
    """
    Ex. 3.1 and 3.2:
    Use W0 = 45 pC, stimulate E population of unit 5 from t = 100 ms,
    plot activity over time and the mean spatial activity profile after onset.
    """

    print("Running Ex. 3.1 and Ex. 3.2...")

    T = 200.0
    n_steps = int(T / DT)
    stimulus_onset = 100.0
    onset_index = int(stimulus_onset / DT)

    W_unit, positions = generate_unit_connectivity(
        Nunits=NUNITS,
        sigma=SIGMA,
        W0=W0_EX3,
        g=G_DEFAULT,
        gamma=GAMMA,
    )

    external_E = np.zeros((n_steps, NUNITS))
    external_I = np.zeros((n_steps, NUNITS))
    external_E[onset_index:, TARGET_UNIT] = I0_STIM

    result = simulate_field_model(
        T=T,
        W_unit=W_unit,
        seed=300,
        external_E=external_E,
        external_I=external_I,
    )

    times = result["times"]
    rates_E = result["rates_E_hz"]
    rates_I = result["rates_I_hz"]

    plot_activity_heatmap(
        times,
        rates_E,
        filename="ex3_1_step_stimulus_unit5_W0_45.png",
        title="Ex. 3.1 - Step stimulus to E population of unit 5, W0 = 45 pC",
        stimulus_unit=TARGET_UNIT,
        stimulus_onset=stimulus_onset,
    )

    # Mean response after stimulus onset. This is the spatial response profile.
    mean_E_after_onset = rates_E[onset_index:].mean(axis=0)
    mean_I_after_onset = rates_I[onset_index:].mean(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(NUNITS), mean_E_after_onset, "o-", label="Excitatory")
    plt.plot(np.arange(NUNITS), mean_I_after_onset, "s--", label="Inhibitory")
    plt.axvline(TARGET_UNIT, linestyle=":", label="Stimulated unit")
    plt.xlabel("E-I unit id")
    plt.ylabel("Mean firing rate after stimulus onset (Hz)")
    plt.title("Ex. 3.2 - Spatial response profile / tuning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex3_2_spatial_response_profile.png"), dpi=200)
    plt.close()

    print("Mean E activity after onset by unit:", np.round(mean_E_after_onset, 2))
    print("Mean I activity after onset by unit:", np.round(mean_I_after_onset, 2))
    print("Ex. 3.1 and Ex. 3.2 figures saved.")

    return mean_E_after_onset, mean_I_after_onset


def exercise_3_3_to_3_5():
    """
    Ex. 3.3 to Ex. 3.5:
    Apply Gaussian stimuli of different widths centered on the target unit.
    Measure the mean rate of the target E population during the last 150 ms.
    Also track the target I population and target field inputs to identify the
    direct cause of surround suppression.
    """

    print("Running Ex. 3.3, Ex. 3.4 and Ex. 3.5...")

    T = 200.0
    n_steps = int(T / DT)
    analysis_start = int(50.0 / DT)  # last 150 ms

    W_unit, positions = generate_unit_connectivity(
        Nunits=NUNITS,
        sigma=SIGMA,
        W0=W0_EX3,
        g=G_DEFAULT,
        gamma=GAMMA,
    )

    center_position = positions[TARGET_UNIT]
    sigma_values = np.round(np.arange(0.0, 0.401, 0.04), 2)

    target_E_rates = []
    target_I_rates = []
    target_field_E = []
    target_field_I = []
    all_stim_profiles = []

    for idx, sigma_stim in enumerate(sigma_values):
        stim_profile = gaussian_stimulus_profile(
            positions=positions,
            center_position=center_position,
            sigma_stim=sigma_stim,
            I0=I0_STIM,
        )
        all_stim_profiles.append(stim_profile)

        external_E = np.tile(stim_profile, (n_steps, 1))
        external_I = np.zeros((n_steps, NUNITS))

        result = simulate_field_model(
            T=T,
            W_unit=W_unit,
            seed=400 + idx * 17,
            external_E=external_E,
            external_I=external_I,
        )

        rates_E = result["rates_E_hz"]
        rates_I = result["rates_I_hz"]
        field_E = result["field_inputs_E"]
        field_I = result["field_inputs_I"]

        target_E_rates.append(rates_E[analysis_start:, TARGET_UNIT].mean())
        target_I_rates.append(rates_I[analysis_start:, TARGET_UNIT].mean())
        target_field_E.append(field_E[analysis_start:, TARGET_UNIT].mean())
        target_field_I.append(field_I[analysis_start:, TARGET_UNIT].mean())

    target_E_rates = np.array(target_E_rates)
    target_I_rates = np.array(target_I_rates)
    target_field_E = np.array(target_field_E)
    target_field_I = np.array(target_field_I)
    all_stim_profiles = np.array(all_stim_profiles)

    peak_index = int(np.argmax(target_E_rates))
    peak_sigma = sigma_values[peak_index]
    peak_rate = target_E_rates[peak_index]

    # Ex. 3.3 main tuning curve: target E response versus stimulus width.
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_values, target_E_rates, "o-", label="Target excitatory population")
    plt.axvline(peak_sigma, linestyle="--", label=f"Peak at sigma = {peak_sigma:.2f}")
    plt.xlabel("Stimulus width sigma_stim")
    plt.ylabel("Mean firing rate in last 150 ms (Hz)")
    plt.title("Ex. 3.3 - Surround suppression tuning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex3_3_width_tuning_target_E.png"), dpi=200)
    plt.close()

    # Diagnostic plot for Ex. 3.5: compare E and I rate of target unit.
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_values, target_E_rates, "o-", label="Target E rate")
    plt.plot(sigma_values, target_I_rates, "s--", label="Target I rate")
    plt.axvline(SIGMA, linestyle=":", label="Network sigma = 0.2")
    plt.xlabel("Stimulus width sigma_stim")
    plt.ylabel("Mean firing rate in last 150 ms (Hz)")
    plt.title("Ex. 3.5 - Target E and I rates versus stimulus width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex3_5_target_EI_rates_vs_width.png"), dpi=200)
    plt.close()

    # Diagnostic plot: field input into target E and target I populations.
    # Target E receives short-range excitation. Target I receives far-range excitation.
    plt.figure(figsize=(8, 5))
    plt.plot(sigma_values, target_field_E, "o-", label="Field input to target E")
    plt.plot(sigma_values, target_field_I, "s--", label="Field input to target I")
    plt.axvline(SIGMA, linestyle=":", label="Network sigma = 0.2")
    plt.xlabel("Stimulus width sigma_stim")
    plt.ylabel("Mean field current in last 150 ms (nA)")
    plt.title("Ex. 3.5 - Field inputs into the investigated unit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex3_5_field_inputs_vs_width.png"), dpi=200)
    plt.close()

    # Optional visualization of the stimulus profiles used.
    plt.figure(figsize=(8, 5))
    for selected_sigma in [0.0, 0.08, 0.16, 0.24, 0.32, 0.40]:
        idx = int(np.where(np.isclose(sigma_values, selected_sigma))[0][0])
        plt.plot(np.arange(NUNITS), all_stim_profiles[idx], "o-", label=f"sigma={selected_sigma:.2f}")
    plt.axvline(TARGET_UNIT, linestyle=":", label="Target unit")
    plt.xlabel("E-I unit id")
    plt.ylabel("External input to E population (nA)")
    plt.title("Ex. 3.3 - Examples of stimulus profiles")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "ex3_3_stimulus_profiles.png"), dpi=200)
    plt.close()

    print("sigma_stim values:", sigma_values)
    print("Target E rates:", np.round(target_E_rates, 2))
    print("Target I rates:", np.round(target_I_rates, 2))
    print("Field input to target E:", np.round(target_field_E, 2))
    print("Field input to target I:", np.round(target_field_I, 2))
    print(f"Peak target E response at sigma_stim = {peak_sigma:.2f}, rate = {peak_rate:.2f} Hz")

    if target_E_rates[-1] < peak_rate:
        print("Surround suppression observed: the target E rate decreases for wide stimuli.")
    else:
        print("Warning: clear surround suppression was not observed in this run. Try another random seed or inspect the plots.")

    if target_I_rates[-1] > target_I_rates[peak_index]:
        print("Diagnostic: target I firing increases for wide stimuli, consistent with increased local inhibition as the direct cause.")
    else:
        print("Diagnostic: target I firing did not increase clearly; inspect field input plots and consider stochastic variability.")

    print("Ex. 3.3, Ex. 3.4 and Ex. 3.5 figures saved.")

    return {
        "sigma_values": sigma_values,
        "target_E_rates": target_E_rates,
        "target_I_rates": target_I_rates,
        "target_field_E": target_field_E,
        "target_field_I": target_field_I,
        "peak_sigma": peak_sigma,
    }


def main():
    make_output_folder()

    print("Exercise 3 - Paradoxical response properties")
    print("Parameters:")
    print(f"NUNITS = {NUNITS}")
    print(f"NE = {NE}, NI = {NI}")
    print(f"network sigma = {SIGMA}")
    print(f"W0 = {W0_EX3} pC")
    print(f"stimulus strength I0 = {I0_STIM} nA")
    print(f"target unit = {TARGET_UNIT}")
    print(f"delay steps = {DELAY_STEPS}")

    exercise_3_1_and_3_2()
    width_results = exercise_3_3_to_3_5()

    print("All Ex. 3 simulations finished.")
    print("Summary for report:")
    print(f"- W0 = {W0_EX3} pC is below the spontaneous bump threshold estimated in Ex. 2.")
    print(f"- The target E response peaked at sigma_stim = {width_results['peak_sigma']:.2f}.")
    print("- For wider stimuli, distal E populations recruit I populations and can suppress the target E response.")


if __name__ == "__main__":
    main()
