import numpy as np
from scipy.sparse import csr_matrix


N_E= 1000
gamma = 0.25
N_I = int(gamma * N_E)
p = 0.02
g = 5
J = 45
tau_delay = 2



def generate_sparse_connectivity(NE, NI, KE, KI, J, g):
    N = NE + NI
    w = np.zeros((N, N))

    exc_neurons = np.arange(NE)
    inh_neurons = np.arange(NE, N)

    for i in range(N):
        valid_exc = exc_neurons[exc_neurons != i]
        valid_inh = inh_neurons[inh_neurons != i]

        exc_connections = np.random.choice(valid_exc, size = KE, replace = False)
        inh_connections = np.random.choice(valid_inh, size = KI, replace = False)

        w[i, exc_connections] = J 
        w[i, inh_connections] = -g *J 

    w = csr_matrix(w)
    return w




def main():
    K_E = int(p * N_E)
    K_I = int(p * N_I)

    w = generate_sparse_connectivity(N_E, N_I, K_E, K_I, J, g)
    print(w)

if __name__ == "__main__": 
    main()