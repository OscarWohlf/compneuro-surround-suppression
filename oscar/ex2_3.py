import numpy as np
from scipy.sparse import csr_matrix

def calc_dist(x,y):
    return min(np.abs(x-y), 1 - np.abs(x-y))

def generate_unit_connectivity(Nunits, sigma, W0, g, gamma):
    W = np.zeros((2 * Nunits, 2 * Nunits))
    spacing = 1 / Nunits

    for i in range(Nunits):
        curr_loc = i * spacing 
        for j in range(Nunits):
            new_loc = j * spacing 
            if i == j:
                continue 
            dist = calc_dist(curr_loc, new_loc)

            f = int(dist <= (sigma + 1e-12))

            W[i,j] = W0 * f 
            W[i + Nunits, j] = g * gamma * W0 * (1 - f)

    W = csr_matrix(W)
    return W

def main():
    Nunits = 10
    sigma = 0.2
    gamma = 0.25
    g = 5
    W0 = 90
    W = generate_unit_connectivity(Nunits, sigma, W0, g, gamma)

if __name__ == "__main__": 
    main()