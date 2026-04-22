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
    return I_0 * (1 + np.sin(omega * t / 1000))

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

def rasterplot(spikes, delta_t):
    spike_times, neuron_ids = np.where(spikes > 0)
    times_ms = spike_times * delta_t

    plt.figure(figsize=(10, 6))
    plt.scatter(times_ms, neuron_ids, s=5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    plt.title("Raster plot of spike times")
    plt.show()

def mean_firing_rate(spikes):
    spike_counts = np.sum(spikes > 0, axis=1)
    mean_rate = spike_counts / N 
    times = np.arange(0,T,delta_t)
    input_current = oscillating_input(I_0, omega, times)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(times, mean_rate)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Mean firing activity")
    axes[0].set_title("Mean firing rate")

    axes[1].plot(times, input_current)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Input current")
    axes[1].set_title("External input")

    plt.tight_layout()
    plt.show()

def main():
    neurons = init_neurons(N, u_reset, theta)
    potentials, spikes = membrane_evolution(neurons)
    rasterplot(spikes, delta_t)
    mean_firing_rate(spikes)



if __name__ == "__main__": 
    main()