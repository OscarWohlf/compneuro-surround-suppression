from ex3_1 import *

import numpy as np
import matplotlib.pyplot as plt

def plot_mean_excitatory_activity_ex32(r_history, delta_t, N_units, stim_onset=100):
    # Extract excitatory population activities
    r_E = r_history[:, :N_units]

    # Convert stimulus onset time to step index
    onset_step = int(stim_onset / delta_t)

    # Average activity after stimulus onset
    mean_activity = r_E[onset_step:, :].mean(axis=0)

    # Convert from spikes/ms to Hz
    mean_activity_hz = mean_activity * 1000

    unit_ids = np.arange(N_units)

    plt.figure(figsize=(8, 4))
    plt.plot(unit_ids, mean_activity_hz, marker="o")
    plt.xlabel("Unit id")
    plt.ylabel("Mean excitatory activity after stimulus onset (Hz)")
    plt.title("Mean excitatory activity as a function of unit location")
    plt.xticks(unit_ids)
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    T = 200
    Nunits = 10
    sigma = 0.2
    gamma = 0.25
    g = 5
    W0 = 45
    W = generate_unit_connectivity(Nunits, sigma, W0, g, gamma)
    I_0 = 30
    stim_unit = 5

    units, r_history = simulate_cortical_sheet_ex31(Nunits, W, T, I_0, stim_unit)

    plot_mean_excitatory_activity_ex32(
        r_history,
        delta_t=0.5,
        N_units=Nunits,
        stim_onset=100
    )

if __name__ == "__main__": 
    main()