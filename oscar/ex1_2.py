from ex0_1 import *
from ex0_2 import background_current, avg_mean_firing
from ex1_1 import *
from tqdm import tqdm

n_bg = 25
T = 100
N_E= 1000
gamma = 0.25
N_I = int(gamma * N_E)
p = 0.02
g = 5
J = 45
tau_delay = 2
delta_t = 0.5

def membrane_evolution_ex_12(init_potentials, I_0, w, N):
    n_steps = int(T / delta_t)
    potentials = np.zeros((n_steps + 1, N))
    spikes = np.zeros((n_steps, N))
    potentials[0] = init_potentials
    curr_potentials = init_potentials.copy()

    curr_t = 0
    curr_step = 0

    while curr_t < T:
        I = total_input_currents(I_0, n_bg, w, spikes, tau_delay, curr_step, N)
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


def synaptic_input(t, tau_delay, w, spikes):
    I_syn = np.zeros(w.shape[0])
    delay_steps = int(tau_delay / delta_t)
    spikes_idx = t - delay_steps
    if spikes_idx < 0:
        return np.zeros(w.shape[0])
    
    I_syn = w @ spikes[spikes_idx]
    return I_syn


def total_input_currents(I_0, n_bg, w, spikes, tau_delay, t, N):
    background = background_current(n_bg, N)
    synaptic = synaptic_input(t, tau_delay, w, spikes)
    return synaptic + background + I_0


def main():
    I_0_range = np.arange(-10,51,5)
    mean_f_rates_exc = np.zeros(len(I_0_range))
    mean_f_rates_inh = np.zeros(len(I_0_range))

    K_E = int(p * N_E)
    K_I = int(p * N_I)
    N = N_E + N_I

    w = generate_sparse_connectivity(N_E, N_I, K_E, K_I, J, g)
    for idx, I_0 in enumerate(tqdm(I_0_range)):
        neurons = init_neurons(N, u_reset, theta)
        potentials, spikes = membrane_evolution_ex_12(neurons, I_0, w, N)

        spikes_exc = spikes[:, :N_E]
        spikes_inh = spikes[:, N_E:N_E+N_I]
        avg_rates_exc = avg_mean_firing(spikes_exc, N_E)
        avg_rates_inh = avg_mean_firing(spikes_inh, N_I)
        mean_f_rates_exc[idx] = avg_rates_exc
        mean_f_rates_inh[idx] = avg_rates_inh

    plt.figure(figsize=(10, 6))
    plt.plot(I_0_range, mean_f_rates_exc, label="Excitatory neurons")
    plt.plot(I_0_range, mean_f_rates_inh, label="Inhibatory neurons")
    plt.xlabel("External current I_0 (nA)")
    plt.ylabel("Mean firing rate (Hz)")
    plt.title("f-I curve")
    plt.legend()
    plt.show()
    


if __name__ == "__main__": 
    main()