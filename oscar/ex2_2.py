from ex1_1 import generate_sparse_connectivity
from ex0_1 import init_neurons, delta_u, R, tau_m, u_reset, theta
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

T = 1000
I_0 = 10
omega = 25
n_bg = 25
N_E= 1000
gamma = 0.25
N_I = int(gamma * N_E)
p = 0.02
g = 5
J = 45
tau_delay = 2
delta_t = 0.5


class E_I_unit:
    def __init__(self, N_E, N_I, J, p, tau_delay, delta_t, T, n_bg, g, u_reset, theta):
        self.N_E = N_E
        self.N_I = N_I
        self.N = N_E + N_I
        self.K_E = int(p * N_E)
        self.K_I = int(p * N_I)
        self.p = p
        self.J = J
        self.g = g
        self.n_bg = n_bg
        self.tau_delay = tau_delay
        self.delta_t = delta_t
        self.theta = theta
        self.u_reset = u_reset

        self.w = generate_sparse_connectivity(self.N_E, self.N_I, self.K_E, self.K_I, self.J, self.g)
        self.u = init_neurons(self.N, self.u_reset, self.theta)
        self.n_steps = int(T / delta_t)
        self.potentials = np.zeros((self.n_steps + 1, self.N))
        self.spikes = np.zeros((self.n_steps, self.N))
        self.potentials[0] = self.u.copy()
        self.curr_t = 0 
        self.curr_step = 0

    def local_synaptic_input(self):
        I_syn = np.zeros(self.w.shape[0])
        delay_steps = int(self.tau_delay / self.delta_t)
        spikes_idx = self.curr_step - delay_steps
        if spikes_idx < 0:
            return np.zeros(self.w.shape[0])
        
        I_syn = self.w @ self.spikes[spikes_idx]
        return I_syn

    def local_background_input(self):
        background_input = np.random.poisson(self.n_bg, self.N)
        return background_input
    
    def external_input(self, I_E_ext, I_I_ext):
        I_ext = np.zeros(self.N)
        I_ext[:self.N_E] = I_E_ext
        I_ext[self.N_E:] = I_I_ext
        return I_ext
    
    def total_current(self, I_E_ext, I_I_ext):
        I_syn = self.local_synaptic_input()
        I_bg = self.local_background_input()
        I_ext = self.external_input(I_E_ext, I_I_ext)
        return I_syn + I_bg + I_ext

    def step(self, I_E, I_I):
        I_total = self.total_current(I_E, I_I)

        d_u = delta_u(tau_m, self.u, R, I_total)
        next_potentials = self.u + self.delta_t * d_u

        spiked = next_potentials >= self.theta
        self.spikes[self.curr_step, spiked] = 1 / self.delta_t
        next_potentials[spiked] = self.u_reset

        self.potentials[self.curr_step+ 1] = next_potentials
        self.u = next_potentials

        r_E = np.mean(self.spikes[self.curr_step, :self.N_E])
        r_I = np.mean(self.spikes[self.curr_step, self.N_E:])

        self.curr_t += self.delta_t
        self.curr_step += 1

        return r_E, r_I

def delayed_activity(r_history, curr_step, tau_delay, delta_t):
    N = r_history.shape[1]
    delay_steps = int(tau_delay / delta_t)
    if curr_step >= delay_steps:
        return r_history[curr_step - delay_steps]
    else:
        return np.zeros(r_history.shape[1])

def total_inputs(W, r_delayed):
    N = len(r_delayed) // 2
    I_unit = W @ r_delayed
    I_E_ext = I_unit[:N]
    I_I_ext = I_unit[N:]
    return I_E_ext, I_I_ext

def simulate_cortical_sheet(N_units, W, T):
    n_steps = int(T / delta_t)
    r_history = np.zeros((n_steps, 2 * N_units))
    
    units = [E_I_unit(N_E=N_E,N_I=N_I,J=J,p=p,tau_delay=tau_delay,delta_t=delta_t,T=T,n_bg=n_bg,g=g,u_reset=u_reset,theta=theta)for _ in range(N_units)]

    for curr_step in tqdm(range(n_steps)):
        r_delayed = delayed_activity(r_history, curr_step, tau_delay, delta_t)

        I_E_total_ext, I_I_total_ext = total_inputs(W, r_delayed)
        for unit in range(N_units):
            I_E_total_ext_unit = I_E_total_ext[unit]
            I_I_total_ext_unit = I_I_total_ext[unit]

            r_E, r_I = units[unit].step(I_E=I_E_total_ext_unit, I_I=I_I_total_ext_unit)

            r_history[curr_step, unit] = r_E
            r_history[curr_step, N_units + unit] = r_I

    return units, r_history


def plot_specific_units(r_history, delta_t, N_units, units_to_plot, bin_ms=20):
    steps_per_bin = int(bin_ms / delta_t)
    n_bins = r_history.shape[0] // steps_per_bin

    r_trimmed = r_history[:n_bins * steps_per_bin]
    r_binned = r_trimmed.reshape(n_bins, steps_per_bin, 2 * N_units).mean(axis=1)

    r_binned_hz = r_binned * 1000

    times = np.arange(n_bins) * bin_ms + bin_ms / 2

    for unit_idx in units_to_plot:
        r_E = r_binned_hz[:, unit_idx]
        r_I = r_binned_hz[:, N_units + unit_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(times, r_E, label=f"Unit {unit_idx} excitatory")
        plt.plot(times, r_I, label=f"Unit {unit_idx} inhibitory")
        plt.xlabel("Time (ms)")
        plt.ylabel("Population firing rate (Hz)")
        plt.title(f"Balance check for unit {unit_idx}")
        plt.legend()
        plt.show()

def main():
    N_units = 10
    W = np.zeros((2 * N_units, 2 * N_units))

    units, r_history = simulate_cortical_sheet(N_units, W, T=200)

    plot_specific_units(
        r_history,
        delta_t,
        N_units,
        units_to_plot=[0, 5, 9],
        bin_ms=20
    )

if __name__ == "__main__": 
    main()