"""
Quantum Entanglement Demonstrator (QED)
Density matrix computation
"""
import numpy as np

pauli_matrices = [
            [[1, 0],    # sigma 0
             [0, 1]],
            [[0, 1],    # sigma 1
             [1, 0]],
            [[0, -1j],  # sigma 2
             [1j, 0]],
            [[1, 0],    # sigma 3
             [0, -1]]]

counts = np.array([
    [0.2706599121, 0.2489901582, 0.01295224313,
     0.5607573565, 0.190942375, 0.341451767],
    [0.2091097771, 0.2712401526, 0.4134346358,
     0.01285576459, 0.1918974849, 0.2757083731],
    [0.01298180766, 0.5107662161, 0.2033958849,
     0.3017003239, 0.177497304, 0.3408555338],
    [0.4601082188, 0.01614375738, 0.2367117546,
     0.2581920366, 0.2064145065, 0.2752326557],
    [0.217271873, 0.3028842678, 0.1796310115,
     0.3209222139, 0.3987591653, 0.01486584482],
    [0.2671501892, 0.21269367, 0.2482566966,
     0.251190078, 0.01176375796, 0.574611232]])

T = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        T[i, j] = counts[2*i, 2*j] - counts[2*i+1, 2*j] - \
                counts[2*i, 2*j+1] + counts[2*i+1, 2*j+1]

print('\n T-matrix:\n', T, '\n')
# Construct the density matrix
density_matrix = np.zeros((4, 4), dtype=np.complex128)
for i in range(3):
    for j in range(3):
        density_matrix += T[i, j] * \
                np.kron(pauli_matrices[i+1], pauli_matrices[j+1])

# Now add the non-full correlations T_{0,x} and T_{x,0} for
# x in [1, 2, 3]
# Coeff T_{0,x}
T_par = np.zeros((4, 4))
T_par[0, 0] = 1
for i in range(3):
    for j in range(3):
        T_par[0, i+1] += counts[2*j, 2*i] - counts[2*j+1, 2*i] + \
                counts[2*j, 2*i+1] - counts[2*j+1, 2*i+1]
    T_par[0, i+1] *= 1/3

for i in range(3):
    for j in range(3):
        T_par[i+1, 0] += counts[2*i, 2*j] + counts[2*i, 2*j+1] - \
                counts[2*i+1, 2*j] - counts[2*i+1, 2*j+1]
    T_par[i+1, 0] *= 1/3

print('T_par:\n', T_par, '\n')

# Add the non-full correlations to the denstiy matrix
density_matrix += np.kron(pauli_matrices[0], pauli_matrices[0])
for i in range(3):
    density_matrix += T_par[0, i+1] * \
        np.kron(pauli_matrices[0], pauli_matrices[i+1])

    density_matrix += T_par[i+1, 0] * \
        np.kron(pauli_matrices[i+1], pauli_matrices[0])

density_matrix *= 0.25

np.set_printoptions(precision=5, linewidth=1000, suppress=True)
print('\n')
print('Density matrix:\n', density_matrix, '\n')
