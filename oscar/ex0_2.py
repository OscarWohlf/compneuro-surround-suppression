from ex0_1 import *
from tqdm import tqdm

n_bg = 25
T = 100

def background_current(n_bg):
    background_input = np.random.poisson(n_bg, N)
    return background_input

def total_input_currents(I_0, n_bg):
    background = background_current(n_bg)
    return background + I_0

def membrane_evolution_ex_02(init_potentials, I_0):
    n_steps = int(T / delta_t)
    potentials = np.zeros((n_steps + 1, N))
    spikes = np.zeros((n_steps, N))
    potentials[0] = init_potentials
    curr_potentials = init_potentials.copy()

    curr_t = 0
    curr_step = 0

    while curr_t < T:
        I = total_input_currents(I_0, n_bg)
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

def avg_mean_firing(spikes):
    spike_counts = np.sum(spikes > 0, axis=1)
    mean_counts = spike_counts / N 
    n_last_steps = int(50 / delta_t)
    avg_mean_counts = np.mean(mean_counts[-n_last_steps:])
    avg_mean_rate = avg_mean_counts / 0.0005 # divide to make it a rate in Hz
    return avg_mean_rate

def theoretical_f_I_curve(I_0_vals):
    theory_rates = np.zeros(len(I_0_vals))

    for idx, I_0 in enumerate(I_0_vals):
        if R * I_0 <= theta:
            theory_rates[idx] = 0
        else:
            theory_rates[idx] = 1000 / (tau_m * np.log((R * I_0 - u_reset) / (R * I_0 - (theta))))

    return theory_rates

def main():
    I_0_range = np.arange(-10,51,5)
    mean_f_rates = np.zeros(len(I_0_range))

    for idx, I_0 in enumerate(tqdm(I_0_range)):
        neurons = init_neurons(N, u_reset, theta)
        potentials, spikes = membrane_evolution_ex_02(neurons, I_0)
        avg_rates = avg_mean_firing(spikes)
        mean_f_rates[idx] = avg_rates

    theory_rates = theoretical_f_I_curve(I_0_range)

    plt.figure(figsize=(10, 6))
    plt.plot(I_0_range, mean_f_rates, label="Simulated with background noise")
    plt.plot(I_0_range, theory_rates,label="Theoretical noiseless LIF")
    plt.xlabel("External current I_0 (nA)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title("f-I curve")
    plt.legend()
    plt.show()
    


if __name__ == "__main__": 
    main()