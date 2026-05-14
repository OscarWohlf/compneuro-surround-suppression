from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.parameters import (
    DT,
    N_E,
    N_I,
    K_E,
    K_I,
    J,
    G,
    GAMMA,
)
from src.lif import oscillating_input
from src.connectivity import generate_sparse_connectivity
from src.ei_network import simulate_ei_network, bin_population_rate
from src.plotting import (
    plot_fi_curves,
    plot_ei_raster,
    plot_two_population_rates,
    save_or_show
)

FIGURE_DIR = "figures/ex1"

def exercise_1_2(seed=10, save_figures=True):
    rng = np.random.default_rng(seed)

    w = generate_sparse_connectivity(rng=rng)

    t_max = 100.0
    i_values = np.arange(-10.0, 55.0, 5.0)

    rates_e = np.zeros(len(i_values))
    rates_i = np.zeros(len(i_values))

    start_index = int(50.0 / DT)

    for idx, i0 in enumerate(tqdm(i_values)):
        sim_rng = np.random.default_rng(seed + idx + 100)

        _, _, _, rate_e, rate_i = simulate_ei_network(
            t_max=t_max,
            w=w,
            i_ext_e=i0,
            i_ext_i=i0,
            rng=sim_rng,
        )

        rates_e[idx] = np.mean(rate_e[start_index:])
        rates_i[idx] = np.mean(rate_i[start_index:])

    path = f"{FIGURE_DIR}/ex1_2_fi_curves.png" if save_figures else None
    plot_fi_curves(i_values, rates_e, rates_i, path=path)

    return i_values, rates_e, rates_i

def exercise_1_4(seed=20, save_figures=True):
    g_values = [1.0, 2.0, 4.0, 5.0]

    t_max = 300.0
    start_index = int(150.0 / DT)

    mean_rates_e = []
    mean_rates_i = []

    for idx, g in enumerate(g_values):
        rng = np.random.default_rng(seed + idx)

        w = generate_sparse_connectivity(g=g, rng=rng)

        _, _, _, rate_e, rate_i = simulate_ei_network(
            t_max=t_max,
            w=w,
            i_ext_e=0.0,
            i_ext_i=0.0,
            rng=rng,
        )

        mean_rates_e.append(np.mean(rate_e[start_index:]))
        mean_rates_i.append(np.mean(rate_i[start_index:]))

    x = np.arange(len(g_values))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, mean_rates_e, width, label="Excitatory")
    plt.bar(x + width / 2, mean_rates_i, width, label="Inhibitory")
    plt.axvline(2.5, linestyle="--", label="boundary near $g=4$")
    plt.xticks(x, [str(g) for g in g_values])
    plt.xlabel("Inhibition strength $g$")
    plt.ylabel("Mean firing rate after transient (Hz)")
    plt.title("Effect of changing inhibition strength")
    plt.legend()

    path = f"{FIGURE_DIR}/ex1_4_g_comparison.png" if save_figures else None
    save_or_show(path)

    return np.array(g_values), np.array(mean_rates_e), np.array(mean_rates_i)

def make_ex15_inputs(t_max=1000.0, i0=10.0, omega=25.0):
    n_steps = int(t_max / DT)
    times = np.arange(n_steps) * DT

    osc = oscillating_input(i0, omega, times)

    i_ext_e = np.zeros(n_steps)
    i_ext_i = np.zeros(n_steps)

    first_half = times < 500.0
    second_half = times >= 500.0

    i_ext_e[first_half] = osc[first_half]
    i_ext_i[second_half] = osc[second_half]

    return times, i_ext_e, i_ext_i


def exercise_1_5(seed=30, save_figures=True):
    rng = np.random.default_rng(seed)

    w = generate_sparse_connectivity(rng=rng)

    t_max = 1000.0
    times, i_ext_e, i_ext_i = make_ex15_inputs(t_max=t_max)

    times, potentials, spikes, rate_e, rate_i = simulate_ei_network(
        t_max=t_max,
        w=w,
        i_ext_e=i_ext_e,
        i_ext_i=i_ext_i,
        rng=rng,
    )

    raster_path = f"{FIGURE_DIR}/ex1_5_raster.png" if save_figures else None

    plot_ei_raster(
        spikes,
        n_e=N_E,
        n_e_plot=100,
        n_i_plot=25,
        dt=DT,
        switch_time=500.0,
        path=raster_path,
    )

    bin_times_e, binned_e = bin_population_rate(spikes, slice(0, N_E), bin_ms=20.0)
    bin_times_i, binned_i = bin_population_rate(spikes, slice(N_E, N_E + N_I), bin_ms=20.0)

    rate_path = f"{FIGURE_DIR}/ex1_5_binned_rates.png" if save_figures else None

    plot_two_population_rates(
        bin_times_e,
        binned_e,
        binned_i,
        switch_time=500.0,
        path=rate_path,
    )

    return times, potentials, spikes, rate_e, rate_i, binned_e, binned_i

def main():
    exercise_1_2()
    exercise_1_4()
    exercise_1_5()

if __name__ == "__main__":
    main()