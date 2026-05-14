
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix

from src.parameters import (
    N_E,
    N_I,
    N,
    P,
    J,
    G,
    GAMMA,
    N_BG,
    DT,
    TAU_DELAY,
    THETA,
    U_RESET,
    TAU_M,
    R,
)

from src.lif import init_neurons, delta_u
from src.connectivity import generate_sparse_connectivity

class EIUnit:
    def __init__(self, t_max, n_e=N_E, n_i=N_I, p=P, j=J, g=G, n_bg=N_BG, dt=DT, tau_delay=TAU_DELAY,
        theta=THETA, u_reset=U_RESET, tau_m=TAU_M, res=R, rng=None,):
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.n_e = n_e
        self.n_i = n_i
        self.n = n_e + n_i
        self.k_e = int(p * n_e)
        self.k_i = int(p * n_i)
        self.j = j
        self.g = g
        self.n_bg = n_bg
        self.dt = dt
        self.tau_delay = tau_delay
        self.theta = theta
        self.u_reset = u_reset
        self.tau_m = tau_m
        self.res = res

        self.w = generate_sparse_connectivity(n_e=n_e,n_i=n_i,k_e=self.k_e,k_i=self.k_i,
            j=j,g=g,rng=rng,)
        self.u = init_neurons(self.n, u_reset=u_reset, theta=theta, rng=rng)

        self.n_steps = int(t_max / dt)
        self.potentials = np.zeros((self.n_steps + 1, self.n))
        self.spikes = np.zeros((self.n_steps, self.n))

        self.potentials[0] = self.u.copy()

        self.curr_step = 0


    def local_synaptic_input(self):
        delay_steps = int(self.tau_delay / self.dt)
        delayed_step = self.curr_step - delay_steps

        if delayed_step < 0:
            return np.zeros(self.n)

        return self.w @ self.spikes[delayed_step]


    def local_background_input(self):
        return self.rng.poisson(self.n_bg, size=self.n)

    def external_input(self, i_e_ext, i_i_ext):
        i_ext = np.zeros(self.n)
        i_ext[:self.n_e] = i_e_ext
        i_ext[self.n_e:] = i_i_ext
        return i_ext

    def total_current(self, i_e_ext, i_i_ext):
        i_syn = self.local_synaptic_input()
        i_bg = self.local_background_input()
        i_ext = self.external_input(i_e_ext, i_i_ext)
        return i_syn + i_bg + i_ext

    def step(self, i_e_ext, i_i_ext):
        if self.curr_step >= self.n_steps:
            raise RuntimeError("EIUnit has already reached the end of the simulation.")

        i_total = self.total_current(i_e_ext, i_i_ext)

        du = delta_u(self.tau_m, self.u, self.res, i_total)
        next_u = self.u + self.dt * du

        spiked = next_u >= self.theta
        self.spikes[self.curr_step, spiked] = 1.0 / self.dt
        next_u[spiked] = self.u_reset

        self.potentials[self.curr_step + 1] = next_u
        self.u = next_u

        # These are population activities in 1/ms, because spikes are stored as 1/dt.
        r_e = np.mean(self.spikes[self.curr_step, :self.n_e])
        r_i = np.mean(self.spikes[self.curr_step, self.n_e:])

        self.curr_step += 1

        return r_e, r_i

def calc_dist(x, y):
    return min(abs(x - y), 1.0 - abs(x - y))


def generate_unit_connectivity(n_units, sigma, w0, g=G, gamma=GAMMA):
    W = np.zeros((2 * n_units, 2 * n_units))
    spacing = 1.0 / n_units

    for alpha in range(n_units):
        x_alpha = alpha * spacing

        for beta in range(n_units):
            x_beta = beta * spacing
            dist = calc_dist(x_alpha, x_beta)

            f = int(dist <= sigma + 1e-12)

            if alpha != beta:
                W[alpha, beta] = w0 * f

            # I_alpha receives excitation from far-away E_beta.
            W[n_units + alpha, beta] = g * gamma * w0 * (1 - f)

            # No between-unit connections from inhibitory populations.
            W[alpha, n_units + beta] = 0.0
            W[n_units + alpha, n_units + beta] = 0.0

    return csr_matrix(W)

def delayed_activity(r_history, curr_step, tau_delay=TAU_DELAY, dt=DT):
    delay_steps = int(tau_delay / dt)

    if curr_step < delay_steps:
        return np.zeros(r_history.shape[1])

    return r_history[curr_step - delay_steps]

def total_inputs(W, r_delayed):
    n_units = len(r_delayed) // 2
    i_unit = W @ r_delayed

    i_e_ext = np.asarray(i_unit[:n_units]).reshape(-1)
    i_i_ext = np.asarray(i_unit[n_units:]).reshape(-1)

    return i_e_ext, i_i_ext

def simulate_cortical_sheet(
    n_units,
    W,
    t_max,
    external_e=None,
    external_i=None,
    rng=None,
    record_inputs=False,
):
    if rng is None:
        rng = np.random.default_rng()

    n_steps = int(t_max / DT)
    times = np.arange(n_steps) * DT

    if external_e is None:
        external_e = np.zeros((n_steps, n_units))
    if external_i is None:
        external_i = np.zeros((n_steps, n_units))

    units = [
        EIUnit(t_max=t_max, rng=np.random.default_rng(rng.integers(0, 1_000_000_000)))
        for _ in range(n_units)
    ]

    r_history = np.zeros((n_steps, 2 * n_units))

    if record_inputs:
        field_e_history = np.zeros((n_steps, n_units))
        field_i_history = np.zeros((n_steps, n_units))
    else:
        field_e_history = None
        field_i_history = None

    for curr_step in tqdm(range(n_steps)):
        r_delayed = delayed_activity(r_history, curr_step)

        field_e, field_i = total_inputs(W, r_delayed)

        if record_inputs:
            field_e_history[curr_step] = field_e
            field_i_history[curr_step] = field_i

        for unit_idx, unit in enumerate(units):
            i_e = field_e[unit_idx] + external_e[curr_step, unit_idx]
            i_i = field_i[unit_idx] + external_i[curr_step, unit_idx]

            r_e, r_i = unit.step(i_e_ext=i_e, i_i_ext=i_i)

            r_history[curr_step, unit_idx] = r_e
            r_history[curr_step, n_units + unit_idx] = r_i

    if record_inputs:
        return times, units, r_history, field_e_history, field_i_history

    return times, units, r_history

def unit_positions(n_units):
    return np.arange(n_units) / n_units


def gaussian_stimulus(n_units, center_unit, sigma_stim, i0):
    positions = unit_positions(n_units)
    center_position = positions[center_unit]

    distances = np.array([
        calc_dist(x, center_position)
        for x in positions
    ])

    stimulus = np.zeros(n_units)

    if sigma_stim == 0:
        stimulus[center_unit] = i0
    else:
        stimulus = i0 * np.exp(-(distances ** 2) / (2 * sigma_stim ** 2))

    return stimulus