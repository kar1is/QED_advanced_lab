"""
Quantum Entanglement Demonstrator (QED)
2-Qubit Tomography Statistical Error Calculation
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

thetas_p = [np.pi/4, -np.pi/4, np.pi/2, 0, 0, np.pi/2]
thetas_qwp = [np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0, 0]

counts = np.array([
    [12128, 11157, 537, 23249, 4798, 8580],
    [9370, 12154, 17141, 533, 4822, 6928],
    [583, 22938, 8541, 12669, 4444, 8534],
    [20663, 725, 9940, 10842, 5168, 6891],
    [9017, 12570, 6981, 12472, 9898, 369],
    [11087, 8827, 9648, 9762, 292, 14263]])

counts2 = np.array([
    [6784, 16994, 13323, 1503, 3410, 3461],
    [12616, 9348, 1241, 20601, 2429, 3705],
    [18644, 1464, 9938, 7831, 2769, 3991],
    [1315, 25010, 4398, 14872, 3371, 3284],
    [10120, 11959, 5262, 11763, 5563, 194],
    [6969, 9592, 6732, 8762, 144, 6201]])

# Pick which measurement run to use
# counts = counts2
counts = counts.T


T = np.zeros((3, 3))
T_err = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        T[i, j] = counts[2*i, 2*j] - counts[2*i+1, 2*j] - \
                counts[2*i, 2*j+1] + counts[2*i+1, 2*j+1]
        T_err[i, j] = 4 * ((((counts[2*i+1, 2*j]+counts[2*i, 2*j+1])**2) *
                           (counts[2*i, 2*j] + counts[2*i+1, 2*j+1]) +
                           ((counts[2*i+1, 2*j]+counts[2*i, 2*j+1])) *
                           ((counts[2*i, 2*j] + counts[2*i+1, 2*j+1])**2)) /
                           (counts[2*i, 2*j] + counts[2*i+1, 2*j] +
                            counts[2*i, 2*j+1] + counts[2*i+1, 2*j+1])**4)

# Construct the density matrix
density_matrix = np.zeros((4, 4), dtype=np.complex128)
for i in range(3):
    for j in range(3):
        density_matrix += T_err[i, j] * \
                np.kron(pauli_matrices[i+1], pauli_matrices[j+1])

# Now add the non-full correlations T_{0,x} and T_{x,0} for
# x in [1, 2, 3]
# Coeff T_{0,x}
T_par_err = np.zeros((4, 4))
T_par_err[0, 0] = 0
for i in range(3):
    for j in range(3):
        T_par_err[0, i+1] += ((((counts[2*j, 2*i+1]+counts[2*j+1, 2*i+1])**2) *
                               (counts[2*j, 2*i] + counts[2*j+1, 2*i]) +
                               ((counts[2*j, 2*i]+counts[2*j+1, 2*i])**2) *
                               (counts[2*j, 2*i+1] + counts[2*j+1, 2*i+1])) /
                              (counts[2*j, 2*i] + counts[2*j+1, 2*i] +
                               counts[2*j, 2*i+1] + counts[2*j+1, 2*i+1])**4)
    T_par_err[0, i+1] *= 4/3

for i in range(3):
    for j in range(3):
        T_par_err[i+1, 0] += ((((counts[2*j+1, 2*i]+counts[2*j+1, 2*i+1])**2) *
                               (counts[2*j, 2*i] + counts[2*j, 2*i+1]) +
                               ((counts[2*j, 2*i]+counts[2*j, 2*i+1])**2) *
                               (counts[2*j+1, 2*i] + counts[2*j+1, 2*i+1])) /
                              (counts[2*j, 2*i] + counts[2*j+1, 2*i] +
                               counts[2*j, 2*i+1] + counts[2*j+1, 2*i+1])**4)
    T_par_err[i+1, 0] *= 4/3

# Add the non-full correlations to the denstiy matrix
for i in range(3):
    density_matrix += T_par_err[0, i+1] * \
        np.kron(pauli_matrices[0], pauli_matrices[i+1])

    density_matrix += T_par_err[i+1, 0] * \
        np.kron(pauli_matrices[i+1], pauli_matrices[0])

re = np.real(density_matrix)
re = np.sqrt(np.abs(re))
im = np.imag(density_matrix)
im = np.sqrt(np.abs(im))

density_matrix = re + 1j * im

density_matrix *= 0.25

print(' T_err-matrix:\n', T_err, '\n')
print(' SD matrix:\n', density_matrix, '\n')
