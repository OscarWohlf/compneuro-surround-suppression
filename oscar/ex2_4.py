from ex2_2 import E_I_unit, simulate_cortical_sheet, delayed_activity, total_inputs
from ex2_3 import generate_unit_connectivity
import numpy as np
import matplotlib.pyplot as plt


def plot_excitatory_activity(r_history, T, Nunits, delta_t=0.5, bin_ms=5):
    r_history = np.asarray(r_history)

    # Extract excitatory population activities
    r_E = r_history[:, :Nunits]   # shape: (n_steps, Nunits)

    # Bin activity over time to make the bump visible
    steps_per_bin = int(bin_ms / delta_t)
    n_bins = r_E.shape[0] // steps_per_bin

    r_E_trimmed = r_E[:n_bins * steps_per_bin]
    r_E_binned = r_E_trimmed.reshape(n_bins, steps_per_bin, Nunits).mean(axis=1)

    # Convert from spikes/ms to Hz, since spikes were stored as 1 / delta_t
    r_E_binned_hz = r_E_binned * 1000

    plt.figure(figsize=(9, 4))

    plt.imshow(
        r_E_binned_hz.T,
        aspect="auto",
        origin="lower",
        extent=[0, n_bins * bin_ms, -0.5, Nunits - 0.5],
        interpolation="nearest"
    )

    plt.xlabel("Time (ms)")
    plt.ylabel("Unit id")
    plt.yticks(range(Nunits))
    plt.title("Excitatory population activity over time")
    plt.colorbar(label="Excitatory activity / firing rate (Hz)")

    plt.tight_layout()
    plt.show()

def main():
    T = 200
    Nunits = 10
    sigma = 0.2
    gamma = 0.25
    g = 5
    W0 = 75
    W = generate_unit_connectivity(Nunits, sigma, W0, g, gamma)
    
    units, r_history = simulate_cortical_sheet(Nunits, W, T)

    plot_excitatory_activity(r_history, T, Nunits, delta_t=0.5, bin_ms=5)

if __name__ == "__main__": 
    main()