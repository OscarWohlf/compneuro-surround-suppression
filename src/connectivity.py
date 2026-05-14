import numpy as np
from scipy.sparse import csr_matrix

from src.parameters import N_E, N_I, K_E, K_I, J, G

def generate_sparse_connectivity(n_e=N_E, n_i=N_I, k_e=K_E, k_i=K_I, j=J, g=G, rng=None, allow_self_connections=False,):
    if rng is None:
        rng = np.random.default_rng()

    N = n_e + n_i

    w = np.zeros((N, N))

    exc_neurons = np.arange(n_e)
    inh_neurons = np.arange(n_e, N)

    for post in range(N):
        valid_exc = exc_neurons[exc_neurons != post]
        valid_inh = inh_neurons[inh_neurons != post]

        exc_connections = rng.choice(valid_exc, size = k_e, replace = False)
        inh_connections = rng.choice(valid_inh, size = k_i, replace = False)

        w[post, exc_connections] = j
        w[post, inh_connections] = -g *j 

    w = csr_matrix(w)
    return w