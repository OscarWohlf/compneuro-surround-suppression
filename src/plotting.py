import numpy as np
import matplotlib.pyplot as plt
import os
from src.lif import DT

def save_or_show(path=None):
    plt.tight_layout()
    if path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=200)
        plt.close()


def plot_raster(spikes, dt=DT, path=None):
    spike_steps, neuron_ids = np.where(spikes > 0)
    times_ms = spike_steps * dt

    plt.figure(figsize=(10, 6))
    plt.scatter(times_ms, neuron_ids, s=5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.title("Raster plot of spike times")
    save_or_show(path)


def plot_rate_and_input(times, rate_hz, input_current, path=None):
    input_scaled = input_current / np.max(input_current) * np.max(rate_hz)

    plt.figure(figsize=(10, 5))
    plt.plot(times, rate_hz, label="Mean firing rate")
    plt.plot(times, input_scaled, "--", label="External input, rescaled")
    plt.xlabel("Time (ms)")
    plt.ylabel("Population firing rate (Hz)")
    plt.title("Mean firing rate compared with external input")
    plt.legend()
    save_or_show(path)


def plot_ei_raster(
    spikes,
    n_e,
    n_e_plot=100,
    n_i_plot=25,
    dt=0.5,
    switch_time=None,
    path=None,
):
    spikes_e = spikes[:, :n_e_plot]
    spikes_i = spikes[:, n_e:n_e + n_i_plot]

    spike_times_e, neuron_ids_e = np.where(spikes_e > 0)
    spike_times_i, neuron_ids_i = np.where(spikes_i > 0)

    plt.figure(figsize=(10, 5))
    plt.scatter(spike_times_e * dt, neuron_ids_e, s=5, label="Excitatory")
    plt.scatter(spike_times_i * dt, neuron_ids_i + n_e_plot, s=5, label="Inhibitory")

    if switch_time is not None:
        plt.axvline(switch_time, linestyle="--", label="Switch stimulation")

    plt.xlabel("Time (ms)")
    plt.ylabel("Displayed neuron ID")
    plt.title("Raster plot of excitatory and inhibitory neurons")
    plt.legend()
    save_or_show(path)


def plot_fi_curves(i_values, rates_e, rates_i, path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(i_values, rates_e, "o-", label="Excitatory population")
    plt.plot(i_values, rates_i, "s-", label="Inhibitory population")
    plt.xlabel("External current $I_0$ (nA)")
    plt.ylabel("Mean firing rate over last 50 ms (Hz)")
    plt.title("f-I curves for E and I populations")
    plt.legend()
    save_or_show(path)

def plot_two_population_rates(
    times,
    rate_e,
    rate_i,
    switch_time=None,
    path=None,
):
    plt.figure(figsize=(10, 5))
    plt.plot(times, rate_e, label="Excitatory population")
    plt.plot(times, rate_i, label="Inhibitory population")

    if switch_time is not None:
        plt.axvline(switch_time, linestyle="--", label="Switch stimulation")

    plt.xlabel("Time (ms)")
    plt.ylabel("Population firing rate (Hz)")
    plt.title("Excitatory and inhibitory population rates")
    plt.legend()
    save_or_show(path)

def plot_excitatory_activity(r_history, n_units, dt=0.5, bin_ms=5.0, path=None):
    r_history = np.asarray(r_history)

    r_e = r_history[:, :n_units]

    steps_per_bin = int(bin_ms / dt)
    n_bins = r_e.shape[0] // steps_per_bin

    r_e_trimmed = r_e[: n_bins * steps_per_bin]
    r_e_binned = r_e_trimmed.reshape(n_bins, steps_per_bin, n_units).mean(axis=1)

    # r_history is in 1/ms, so multiply by 1000 to get Hz.
    r_e_binned_hz = r_e_binned * 1000.0

    plt.figure(figsize=(9, 4))
    plt.imshow(
        r_e_binned_hz.T,
        aspect="auto",
        origin="lower",
        extent=[0, n_bins * bin_ms, -0.5, n_units - 0.5],
        interpolation="nearest",
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Unit id")
    plt.yticks(range(n_units))
    plt.title("Excitatory population activity over time")
    plt.colorbar(label="Excitatory activity (Hz)")
    save_or_show(path)

def plot_specific_units(r_history, n_units, units_to_plot, dt=0.5, bin_ms=20.0, path=None):
    steps_per_bin = int(bin_ms / dt)
    n_bins = r_history.shape[0] // steps_per_bin

    r_trimmed = r_history[: n_bins * steps_per_bin]
    r_binned = r_trimmed.reshape(n_bins, steps_per_bin, 2 * n_units).mean(axis=1)

    r_binned_hz = r_binned * 1000.0
    times = np.arange(n_bins) * bin_ms + bin_ms / 2

    plt.figure(figsize=(10, 5))

    for unit_idx in units_to_plot:
        plt.plot(times, r_binned_hz[:, unit_idx], label=f"E unit {unit_idx}")
        plt.plot(times, r_binned_hz[:, n_units + unit_idx], "--", label=f"I unit {unit_idx}")

    plt.xlabel("Time (ms)")
    plt.ylabel("Population firing rate (Hz)")
    plt.title("Balance check for selected E-I units")
    plt.legend()
    save_or_show(path)

def plot_spatial_profile(mean_profile, target_unit=None, path=None):
    n_units = len(mean_profile)
    unit_ids = np.arange(n_units)

    plt.figure(figsize=(8, 4))
    plt.plot(unit_ids, mean_profile, marker="o")

    if target_unit is not None:
        plt.axvline(target_unit, linestyle=":", label="Stimulated unit")
        plt.legend()

    plt.xlabel("Unit id")
    plt.ylabel("Mean excitatory activity after stimulus onset (Hz)")
    plt.title("Mean excitatory activity as a function of unit location")
    plt.xticks(unit_ids)
    plt.grid(True, alpha=0.3)
    save_or_show(path)

def plot_width_tuning(sigma_values, mean_rates, path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(sigma_values, mean_rates, marker="o")
    plt.xlabel(r"Stimulus width $\sigma_{\mathrm{stim}}$")
    plt.ylabel("Mean firing rate of target E population (Hz)")
    plt.title("Response as a function of stimulus width")
    plt.grid(True, alpha=0.3)
    save_or_show(path)

def plot_width_diagnostics(sigma_values, target_e_rates, target_i_rates, field_e, field_i, path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sigma_values, target_e_rates, "o-", label="Target E")
    axes[0].plot(sigma_values, target_i_rates, "s--", label="Target I")
    axes[0].set_xlabel(r"Stimulus width $\sigma_{\mathrm{stim}}$")
    axes[0].set_ylabel("Mean firing rate (Hz)")
    axes[0].set_title("Target E/I rates")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sigma_values, field_e, "o-", label="Field input to target E")
    axes[1].plot(sigma_values, field_i, "s--", label="Field input to target I")
    axes[1].set_xlabel(r"Stimulus width $\sigma_{\mathrm{stim}}$")
    axes[1].set_ylabel("Mean field input (nA)")
    axes[1].set_title("Field inputs to target unit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    save_or_show(path)