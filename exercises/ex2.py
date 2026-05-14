from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.parameters import N_UNITS, SIGMA, W0_DEFAULT, G, GAMMA, DT
from src.field_model import generate_unit_connectivity, simulate_cortical_sheet
from src.plotting import plot_excitatory_activity, plot_specific_units, save_or_show

FIGURE_DIR = "figures/ex2"

def exercise_2_2(seed=1, save_figures=True):
    W = np.zeros((2 * N_UNITS, 2 * N_UNITS))

    rng = np.random.default_rng(seed)
    times, units, r_history = simulate_cortical_sheet(
        n_units=N_UNITS,
        W=W,
        t_max=200.0,
        rng=rng,
    )

    path = f"{FIGURE_DIR}/ex2_2_balance_check.png" if save_figures else None
    plot_specific_units(
        r_history,
        n_units=N_UNITS,
        units_to_plot=[0, 5, 9],
        dt=DT,
        bin_ms=20.0,
        path=path,
    )

    return times, units, r_history

def exercise_2_3():
    W = generate_unit_connectivity(
        n_units=N_UNITS,
        sigma=SIGMA,
        w0=W0_DEFAULT,
        g=G,
        gamma=GAMMA,
    )

    return W

def exercise_2_4(seed=2, save_figures=True):
    W = generate_unit_connectivity(
        n_units=N_UNITS,
        sigma=SIGMA,
        w0=W0_DEFAULT,
        g=G,
        gamma=GAMMA,
    )

    rng = np.random.default_rng(seed)
    times, units, r_history = simulate_cortical_sheet(
        n_units=N_UNITS,
        W=W,
        t_max=200.0,
        rng=rng,
    )

    path = f"{FIGURE_DIR}/ex2_4_spontaneous_bump.png" if save_figures else None
    plot_excitatory_activity(
        r_history,
        n_units=N_UNITS,
        dt=DT,
        bin_ms=5.0,
        path=path,
    )

    return times, units, r_history


def exercise_2_5(seed=3, save_figures=True):
    W0_values = np.arange(0, 121, 15)

    results = []

    for idx, W0 in enumerate(W0_values):
        W = generate_unit_connectivity(
            n_units=N_UNITS,
            sigma=SIGMA,
            w0=W0,
            g=G,
            gamma=GAMMA,
        )

        rng = np.random.default_rng(seed + idx)
        times, units, r_history = simulate_cortical_sheet(
            n_units=N_UNITS,
            W=W,
            t_max=200.0,
            rng=rng,
        )

        r_e_hz = r_history[:, :N_UNITS] * 1000.0
        mean_profile = r_e_hz[int(100 / DT):].mean(axis=0)

        peak = mean_profile.max()
        mean = mean_profile.mean()
        contrast = peak / (mean + 1e-12)
        peak_unit = int(np.argmax(mean_profile))

        results.append((W0, peak, mean, contrast, peak_unit))

        if save_figures:
            path = f"{FIGURE_DIR}/ex2_5_W0_{int(W0):03d}.png"
        else:
            path = None

        plot_excitatory_activity(
            r_history,
            n_units=N_UNITS,
            dt=DT,
            bin_ms=5.0,
            path=path,
        )

        print(
            f"W0={W0:5.1f} | peak={peak:6.2f} Hz | "
            f"mean={mean:6.2f} Hz | contrast={contrast:5.2f} | peak unit={peak_unit}"
        )

    return results

def main():
    exercise_2_2()
    exercise_2_3()
    exercise_2_4()
    exercise_2_5()


if __name__ == "__main__":
    main()