import numpy as np
import math 
import matplotlib.pyplot as plt 
import scipy 

theta = 20
u_reset = -10
delta_t = 0.5
tau_m = 20
R = 1
N = 100 
T = 1000
I_0 = 20
omega = 10

def init_neurons(N, u_reset, theta):
    neurons = np.random.uniform(u_reset, theta, N)
    return neurons

def oscillating_input(I_0, omega, t):
    return I_0 * (1 + math.sin(omega * t / 1000))

def delta_u(tau_m, curr_potentials, R, I):
    return (- curr_potentials + R * I) / tau_m

def membrane_evolution(init_potentials):
    n_steps = int(T / delta_t)
    potentials = np.zeros((n_steps + 1, N))
    spikes = np.zeros((n_steps, N))
    potentials[0] = init_potentials
    curr_potentials = init_potentials.copy()

    curr_t = 0
    curr_step = 0

    while curr_t < T:
        I = oscillating_input(I_0, omega, curr_t)
        d_u = delta_u(tau_m, curr_potentials, R, I)

        next_potentials = curr_potentials + delta_t * d_u

        spiked = next_potentials >= theta
        spikes[curr_step, spiked] = 1 / delta_t
        next_potentials[spiked] = u_reset

        potentials[curr_step+ 1] = next_potentials
        curr_potentials = next_potentials

        curr_t += delta_t
        curr_step += 1

    return potentials, spikes

def main():
    neurons = init_neurons(N, u_reset, theta)
    potentials, spikes = membrane_evolution(neurons)



if __name__ == "__main__": 
    main()