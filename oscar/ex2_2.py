from ex1_1 import generate_sparse_connectivity
from ex0_1 import init_neurons, delta_u, R, tau_m
import numpy as np

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
