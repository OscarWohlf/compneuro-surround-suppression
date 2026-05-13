import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.lif import (
    DT,
    N_BG,
    init_neurons,
    oscillating_input,
    simulate_lif_population,
    population_rate_hz,
    mean_rate_last_window,
    theoretical_lif_rate,
)

from src.plotting import (
    plot_raster,
    plot_rate_and_input,
    save_or_show
)

FIGURE_DIR = "figures/ex0"


def exercise_0_1(seed=1, save_figures=True):
    n_neurons = 100
    t_max = 1000.0
    i0 = 20.0
    omega = 10.0

    rng = np.random.default_rng(seed)

    n_steps = int(t_max / DT)
    times = np.arange(n_steps) * DT

    initial_u = init_neurons(n_neurons, rng=rng)
    input_current = oscillating_input(i0, omega, times)

    times, potentials, spikes = simulate_lif_population(
        initial_u=initial_u,
        t_max=t_max,
        external_current=input_current,
        n_bg=None,
        rng=rng,
    )

    raster_path = f"{FIGURE_DIR}/ex0_1_raster.png" if save_figures else None
    rate_path = f"{FIGURE_DIR}/ex0_1_rate_vs_input.png" if save_figures else None

    plot_raster(spikes, path=raster_path)

    rate_hz = population_rate_hz(spikes)
    plot_rate_and_input(times, rate_hz, input_current, path=rate_path)

    return times, potentials, spikes, rate_hz, input_current


def exercise_0_2(seed=2, save_figures=True):
    n_neurons = 100
    t_max = 100.0
    i_values = np.arange(-10.0, 55.0, 5.0)

    rng = np.random.default_rng(seed)
    simulated_rates = np.zeros(len(i_values))

    for idx, i0 in enumerate(tqdm(i_values)):
        initial_u = init_neurons(n_neurons, rng=rng)

        _, _, spikes = simulate_lif_population(
            initial_u=initial_u,
            t_max=t_max,
            external_current=i0,
            n_bg=N_BG,
            rng=rng,
        )

        simulated_rates[idx] = mean_rate_last_window(spikes, window_ms=50.0)

    theory_rates = theoretical_lif_rate(i_values)

    plt.figure(figsize=(8, 5))
    plt.plot(i_values, simulated_rates, "o-", label="Simulation with background input")
    plt.plot(i_values, theory_rates, "--", label="Theory without background input")
    plt.xlabel("External current $I_0$ (nA)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title("f-I curve")
    plt.legend()

    path = f"{FIGURE_DIR}/ex0_2_fi_curve.png" if save_figures else None
    save_or_show(path)

    return i_values, simulated_rates, theory_rates


def main():
    exercise_0_1()
    exercise_0_2()

if __name__ == "__main__":
    main()