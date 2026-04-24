from ex0_1 import *
from ex0_2 import background_current, avg_mean_firing
from ex1_1 import *
from ex1_2 import synaptic_input
from tqdm import tqdm

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

def external_input_exc(I_0, t, omega):
    if t < 500:
        return I_0 * (1 + np.sin(omega * t / 1000))
    else:
        return 0


def external_input_inh(I_0, t, omega):
    if t < 500:
        return 0
    else:
        return I_0 * (1 + np.sin(omega * t / 1000))

def total_input_currents(I_0, n_bg, w, spikes, tau_delay, t, curr_step, N, omega):
    background = background_current(n_bg, N)
    synaptic = synaptic_input(curr_step, tau_delay, w, spikes)
    exc_ext_input = external_input_exc(I_0, t, omega)
    inh_ext_input = external_input_inh(I_0, t, omega)
    tot_ext = np.zeros(N)
    tot_ext[:N_E] = exc_ext_input
    tot_ext[N_E:] = inh_ext_input
    return synaptic + background + tot_ext


def membrane_evolution_ex15(init_potentials, I_0, w, N, omega):
    n_steps = int(T / delta_t)
    potentials = np.zeros((n_steps + 1, N))
    spikes = np.zeros((n_steps, N))
    potentials[0] = init_potentials
    curr_potentials = init_potentials.copy()

    curr_t = 0
    curr_step = 0

    while curr_t < T:
        I = total_input_currents(I_0, n_bg, w, spikes, tau_delay, curr_t, curr_step, N, omega)
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

def rasterplot_ex15(spikes, delta_t, N_E, n_E_plot, n_I_plot):
    spikes_E = spikes[:, :n_E_plot]
    spikes_I = spikes[:, N_E:N_E + n_I_plot]

    spike_times_E, neuron_ids_E = np.where(spikes_E > 0)
    spike_times_I, neuron_ids_I = np.where(spikes_I > 0)

    times_E = spike_times_E * delta_t
    times_I = spike_times_I * delta_t

    neuron_ids_I = neuron_ids_I + n_E_plot

    plt.figure(figsize=(12, 6))
    plt.scatter(times_E, neuron_ids_E, s=5, label="Excitatory")
    plt.scatter(times_I, neuron_ids_I, s=5, label="Inhibitory")
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("Raster plot of 100 excitatory and 25 inhibitory neurons")
    plt.legend()
    plt.show()

def mean_firing_rate_ex15(spikes, delta_t, N, N_E, N_I, bin_ms):
    spikes_binary = spikes > 0
    spikes_E = spikes_binary[:, :N_E]
    spikes_I = spikes_binary[:, N_E:]
    steps_per_bin = int(bin_ms / delta_t)
    n_bins = spikes.shape[0] // steps_per_bin
    times = np.arange(n_bins) * bin_ms + bin_ms / 2
    bin_s = bin_ms / 1000

    spikes_E = spikes_E[:n_bins * steps_per_bin]
    spikes_I = spikes_I[:n_bins * steps_per_bin]
    spikes_E_binned = spikes_E.reshape(n_bins, steps_per_bin, N_E)
    spikes_I_binned = spikes_I.reshape(n_bins, steps_per_bin, N_I)
    spike_counts_E = spikes_E_binned.sum(axis=(1, 2))
    spike_counts_I = spikes_I_binned.sum(axis=(1, 2))

    rate_E = spike_counts_E / (N_E * bin_s)
    rate_I = spike_counts_I / (N_I * bin_s)

    plt.figure(figsize=(12, 5))
    plt.plot(times, rate_E, label="Excitatory")
    plt.plot(times, rate_I, label="Inhibitory")
    plt.xlabel("Time (ms)")
    plt.ylabel("Population firing rate (Hz)")
    plt.title("Binned population firing rates, 20 ms bins")
    plt.legend()
    plt.show()

def main():
    N = N_E + N_I
    K_E = int(p * N_E)
    K_I = int(p * N_I)
    neurons = init_neurons(N, u_reset, theta)
    w = generate_sparse_connectivity(N_E, N_I, K_E, K_I, J, g)
    potentials, spikes = membrane_evolution_ex15(neurons, I_0, w, N, omega)
    rasterplot_ex15(spikes, delta_t, N_E, 100, 25)
    mean_firing_rate_ex15(spikes, delta_t, N, N_E, N_I, 20)



if __name__ == "__main__": 
    main()