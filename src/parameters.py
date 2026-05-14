
# Ex0 parameters
THETA = 20.0
U_RESET = -10.0
DT = 0.5
TAU_M = 20.0
R = 1.0
N_BG = 25.0

# Ex1 parameters
N_E= 1000
GAMMA = 0.25
N_I = int(GAMMA * N_E)
N = N_E + N_I
P = 0.02
K_E = int(P * N_E)
K_I = int(P * N_I)
G = 5
J = 45
TAU_DELAY = 2
DELAY_STEPS = int(TAU_DELAY / DT)

# Ex2 Parameters
N_UNITS = 10
SIGMA = 0.2
W0_DEFAULT = 90.0
W0_NO_BUMP = 45.0

# Ex3 Parameters
W0_EX3 = 45.0
I0_STIM = 30.0
TARGET_UNIT = 5