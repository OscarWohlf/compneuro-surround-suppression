from ex0_1 import init_neurons, delta_u, R, tau_m, u_reset, theta
from ex2_2 import E_I_unit, simulate_cortical_sheet, delayed_activity, total_inputs
from ex2_3 import generate_unit_connectivity
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

def total_inputs_ex31(W, r_delayed, curr_t, I_0, stim_unit):
    N = len(r_delayed) // 2
    I_unit = W @ r_delayed

    I_E_ext = I_unit[:N]
    I_I_ext = I_unit[N:]

    if curr_t >= 100:
        I_E_ext[stim_unit] += I_0

    return I_E_ext, I_I_ext

def simulate_cortical_sheet_ex31(N_units, W, T, I_0, stim_unit):
    n_steps = int(T / delta_t)
    r_history = np.zeros((n_steps, 2 * N_units))
    
    units = [E_I_unit(N_E=N_E,N_I=N_I,J=J,p=p,tau_delay=tau_delay,delta_t=delta_t,T=T,n_bg=n_bg,g=g,u_reset=u_reset,theta=theta)for _ in range(N_units)]

    for curr_step in tqdm(range(n_steps)):
        r_delayed = delayed_activity(r_history, curr_step, tau_delay, delta_t)

        curr_t = curr_step * delta_t
        I_E_total_ext, I_I_total_ext = total_inputs_ex31(W, r_delayed, curr_t, I_0, stim_unit)
        for unit in range(N_units):
            I_E_total_ext_unit = I_E_total_ext[unit]
            I_I_total_ext_unit = I_I_total_ext[unit]

            r_E, r_I = units[unit].step(I_E=I_E_total_ext_unit, I_I=I_I_total_ext_unit)

            r_history[curr_step, unit] = r_E
            r_history[curr_step, N_units + unit] = r_I

    return units, r_history




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

    plot_excitatory_activity(r_history, T, Nunits, delta_t=0.5, bin_ms=5)

if __name__ == "__main__": 
    main()