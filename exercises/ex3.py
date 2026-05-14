from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.parameters import (
    DT,
    N_UNITS,
    SIGMA,
    G,
    GAMMA,
    W0_EX3,
    I0_STIM,
    TARGET_UNIT,
)
from src.field_model import (
    generate_unit_connectivity,
    simulate_cortical_sheet,
    gaussian_stimulus,
)
from src.plotting import (
    plot_excitatory_activity,
    plot_spatial_profile,
    plot_width_tuning,
    plot_width_diagnostics,
)

FIGURE_DIR = "figures/ex3"



def exercise_3_1(seed=1, save_figures=True):
    t_max = 200.0
    stimulus_onset = 100.0

    n_steps = int(t_max / DT)
    onset_step = int(stimulus_onset / DT)

    W = generate_unit_connectivity(
        n_units=N_UNITS,
        sigma=SIGMA,
        w0=W0_EX3,
        g=G,
        gamma=GAMMA,
    )

    external_e = np.zeros((n_steps, N_UNITS))
    external_i = np.zeros((n_steps, N_UNITS))

    external_e[onset_step:, TARGET_UNIT] = I0_STIM

    rng = np.random.default_rng(seed)

    times, units, r_history = simulate_cortical_sheet(
        n_units=N_UNITS,
        W=W,
        t_max=t_max,
        external_e=external_e,
        external_i=external_i,
        rng=rng,
    )

    path = f"{FIGURE_DIR}/ex3_1_step_stimulus.png" if save_figures else None

    plot_excitatory_activity(
        r_history,
        n_units=N_UNITS,
        dt=DT,
        bin_ms=5.0,
        path=path,
    )

    return times, units, r_history


def exercise_3_2(seed=1, save_figures=True):
    times, units, r_history = exercise_3_1(seed=seed, save_figures=False)

    onset_step = int(100.0 / DT)

    r_e = r_history[:, :N_UNITS]
    mean_profile_hz = r_e[onset_step:].mean(axis=0) * 1000.0

    path = f"{FIGURE_DIR}/ex3_2_spatial_profile.png" if save_figures else None

    plot_spatial_profile(
        mean_profile_hz,
        target_unit=TARGET_UNIT,
        path=path,
    )

    return mean_profile_hz

def exercise_3_3(seed=2, save_figures=True):
    t_max = 200.0
    n_steps = int(t_max / DT)
    analysis_start = int(50.0 / DT)

    W = generate_unit_connectivity(
        n_units=N_UNITS,
        sigma=SIGMA,
        w0=W0_EX3,
        g=G,
        gamma=GAMMA,
    )

    sigma_values = np.round(np.arange(0.0, 0.401, 0.04), 2)

    target_e_rates = []
    target_i_rates = []
    target_field_e = []
    target_field_i = []

    for idx, sigma_stim in enumerate(sigma_values):
        stimulus = gaussian_stimulus(
            n_units=N_UNITS,
            center_unit=TARGET_UNIT,
            sigma_stim=sigma_stim,
            i0=I0_STIM,
        )

        external_e = np.tile(stimulus, (n_steps, 1))
        external_i = np.zeros((n_steps, N_UNITS))

        rng = np.random.default_rng(seed + idx)

        times, units, r_history, field_e, field_i = simulate_cortical_sheet(
            n_units=N_UNITS,
            W=W,
            t_max=t_max,
            external_e=external_e,
            external_i=external_i,
            rng=rng,
            record_inputs=True,
        )

        r_e = r_history[:, :N_UNITS]
        r_i = r_history[:, N_UNITS:]

        target_e_rates.append(r_e[analysis_start:, TARGET_UNIT].mean() * 1000.0)
        target_i_rates.append(r_i[analysis_start:, TARGET_UNIT].mean() * 1000.0)

        target_field_e.append(field_e[analysis_start:, TARGET_UNIT].mean())
        target_field_i.append(field_i[analysis_start:, TARGET_UNIT].mean())

    target_e_rates = np.array(target_e_rates)
    target_i_rates = np.array(target_i_rates)
    target_field_e = np.array(target_field_e)
    target_field_i = np.array(target_field_i)

    path = f"{FIGURE_DIR}/ex3_3_width_tuning.png" if save_figures else None

    plot_width_tuning(
        sigma_values,
        target_e_rates,
        path=path,
    )

    diagnostic_path = f"{FIGURE_DIR}/ex3_5_width_diagnostics.png" if save_figures else None

    plot_width_diagnostics(
        sigma_values,
        target_e_rates,
        target_i_rates,
        target_field_e,
        target_field_i,
        path=diagnostic_path,
    )

    peak_idx = int(np.argmax(target_e_rates))

    print("sigma values:", sigma_values)
    print("target E rates:", np.round(target_e_rates, 2))
    print("target I rates:", np.round(target_i_rates, 2))
    print("field input to target E:", np.round(target_field_e, 2))
    print("field input to target I:", np.round(target_field_i, 2))
    print(f"peak response at sigma_stim = {sigma_values[peak_idx]:.2f}")

    return {
        "sigma_values": sigma_values,
        "target_e_rates": target_e_rates,
        "target_i_rates": target_i_rates,
        "target_field_e": target_field_e,
        "target_field_i": target_field_i,
        "peak_sigma": sigma_values[peak_idx],
    }


def main():
    exercise_3_1()
    exercise_3_2()
    exercise_3_3()


if __name__ == "__main__":
    main()