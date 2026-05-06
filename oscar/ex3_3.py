from ex0_1 import init_neurons, delta_u, R, tau_m, u_reset, theta
from ex2_2 import E_I_unit, simulate_cortical_sheet, delayed_activity, total_inputs
from ex2_3 import generate_unit_connectivity, calc_dist
from ex2_4 import plot_excitatory_activity

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

tau_delay = 2
delta_t = 0.5
n_bg = 25
N_E= 1000
gamma = 0.25
N_I = int(gamma * N_E)
p = 0.02
g = 5
J = 45

def external_input(N, I_0, x_0, sig_stim):
    I_ext = np.zeros(N)

    if sig_stim == 0:
        I_ext[x_0] = I_0
        return I_ext
    
    for i in range(N):
        x_i_ring = i / N
        x_0_ring = x_0 / N
        I_ext[i] = I_0 * np.exp(-(calc_dist(x_i_ring, x_0_ring)**2) / (2 * sig_stim**2))

    return I_ext

def total_inputs_ex33(W, r_delayed, I_0, x_0, sig_stim):
    N = len(r_delayed) // 2
    I_unit = W @ r_delayed

    I_E_ext = I_unit[:N]
    I_I_ext = I_unit[N:]

    I_E_exter = external_input(N, I_0, x_0, sig_stim)

    return I_E_ext + I_E_exter, I_I_ext

def simulate_cortical_sheet_ex33(N_units, W, T, I_0, x_0, sig_stim):
    n_steps = int(T / delta_t)
    r_history = np.zeros((n_steps, 2 * N_units))
    
    units = [E_I_unit(N_E=N_E,N_I=N_I,J=J,p=p,tau_delay=tau_delay,delta_t=delta_t,T=T,n_bg=n_bg,g=g,u_reset=u_reset,theta=theta)for _ in range(N_units)]

    for curr_step in tqdm(range(n_steps)):
        r_delayed = delayed_activity(r_history, curr_step, tau_delay, delta_t)

        curr_t = curr_step * delta_t
        I_E_total_ext, I_I_total_ext = total_inputs_ex33(W, r_delayed, I_0, x_0, sig_stim)
        for unit in range(N_units):
            I_E_total_ext_unit = I_E_total_ext[unit]
            I_I_total_ext_unit = I_I_total_ext[unit]

            r_E, r_I = units[unit].step(I_E=I_E_total_ext_unit, I_I=I_I_total_ext_unit)

            r_history[curr_step, unit] = r_E
            r_history[curr_step, N_units + unit] = r_I

    return units, r_history


def calc_mean_rate_final_part(r_history, delta_t, unit_idx, start_calc):
    start_step = int(start_calc / delta_t)

    r_E_unit = r_history[start_step:, unit_idx]

    # Convert to Hz
    return r_E_unit.mean() * 1000

def main():
    T = 200
    Nunits = 10
    sigma = 0.2
    gamma = 0.25
    g = 5
    W0 = 45
    W = generate_unit_connectivity(Nunits, sigma, W0, g, gamma)
    I_0 = 30
    x_0 = 5
    sig_stims = np.arange(0, 0.401, 0.04)
    mean_rates = []

    for sig_stim in sig_stims:
        units, r_history = simulate_cortical_sheet_ex33(Nunits, W, T, I_0, x_0, sig_stim)

        mean_rate = calc_mean_rate_final_part(r_history, delta_t, x_0, 50)

        mean_rates.append(mean_rate)

    plt.figure(figsize=(8, 4))
    plt.plot(sig_stims, mean_rates, marker="o")
    plt.xlabel(r"Stimulus width $\sigma_{\mathrm{stim}}$")
    plt.ylabel("Mean firing rate of unit 5 E population (Hz)")
    plt.title("Response as a function of stimulus width")
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__": 
    main()