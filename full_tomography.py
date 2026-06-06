"""
Quantum Entanglement Demonstrator (QED)
Monte Carlo Simulation of the Setup Used for 2-Qubit Tomography
"""
import numpy as np


# Jones matrix for a linear polarizer with its axis
# at an angle theta w.r.t. to the horizontal
def linear_polarizer(theta):
    return np.array([[np.cos(theta)**2,
                      np.cos(theta)*np.sin(theta)],
                     [np.cos(theta)*np.sin(theta),
                      np.sin(theta)**2]])


# Jones matrix for a quarter waveplate with its axis
# at an angle theta w.r.t. to the horizontal
def qwp(theta):
    return np.array([[np.cos(theta)**2+1j*np.sin(theta)**2,
                      (1-1j)*np.sin(theta)*np.cos(theta)],
                     [(1-1j)*np.sin(theta)*np.cos(theta),
                      np.sin(theta)**2+1j*np.cos(theta)**2]])


# The optical system placed in one arm
# polarizer mtx * QWP mtx
def system(theta_p, theta_qwp):
    mtx_a = linear_polarizer(theta_p[0]).dot(qwp(theta_qwp[0]))
    mtx_b = linear_polarizer(theta_p[1]).dot(qwp(theta_qwp[1]))
    return np.kron(mtx_a, mtx_b)


# State being considered as set up by the laser
# HH + VV
state = np.array([1, 0, 0, -1]) / np.sqrt(2)

"""
# In case one wishes to test if the system describtion is correct
theta_p = np.array([0, 0])
theta_qwp = np.array([0, 0])
exp_sys = system(theta_p, theta_qwp)
output = exp_sys @ state

print('State: ', state, '\n\n')
print('Experimental system: \n', exp_sys, '\n\n')
print('Final state: ', output)
print('Probability: ', np.vdot(output, output).real)
"""

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

counts = np.zeros((6, 6))

# Calculate coincidence proportions (probabilities) for
# different measurement setting in each arm. The obtained
# counts are of the form c_{x,y}, where x,y are of types
# [H, V, P, M, R, L]
max_runs = 10000
reference_density = np.zeros((4, 4), dtype=np.complex128)
error_samples = []

# Maximum error matrix
# error = np.zeros((4, 4), dtype=np.complex128)

for run in range(max_runs):
    for ch1 in range(len(thetas_p)):
        for ch2 in range(len(thetas_p)):
            # Perturbing the angles within 1deg
            if run != 0:
                # Uniform perturbation
                eps1p, eps2p, eps1q, eps2q = np.random.uniform(
                        (-np.pi/180), (np.pi/180), 4)

                # Gaussian
                # eps1p, eps2p, eps1q, eps2q = np.random.normal(
                #        0, np.pi/180/np.sqrt(3), 4)
            else:
                eps1p, eps2p, eps1q, eps2q = 0, 0, 0, 0
            theta_p = np.array([thetas_p[ch1]+eps1p,
                                thetas_p[ch2]+eps2p])
            theta_qwp = np.array([thetas_qwp[ch1]+eps1q,
                                  thetas_qwp[ch2]+eps2q])
            exp_sys = system(theta_p, theta_qwp)
            output = exp_sys @ state
            counts[ch1, ch2] = np.vdot(output, output).real

    counts = counts.T
    # Compute T_{} coefficients
    # Density matrix goes over T_{x,y}, where x,y in [1, 2, 3]
    # corresponding to x, y, z respectively
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j] = counts[2*i, 2*j] - counts[2*i+1, 2*j] - \
                    counts[2*i, 2*j+1] + counts[2*i+1, 2*j+1]

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
            T_par[0, i+1] += counts[2*j, 2*i] + counts[2*j+1, 2*i] - \
                    counts[2*j, 2*i+1] - counts[2*j+1, 2*i+1]
        T_par[0, i+1] *= 1/3

    for i in range(3):
        for j in range(3):
            T_par[i+1, 0] += counts[2*i, 2*j] + counts[2*i, 2*j+1] - \
                    counts[2*i+1, 2*j] - counts[2*i+1, 2*j+1]
        T_par[i+1, 0] *= 1/3

    # Add the non-full correlations to the denstiy matrix
    density_matrix += np.kron(pauli_matrices[0], pauli_matrices[0])
    for i in range(3):
        density_matrix += T_par[0, i+1] * \
            np.kron(pauli_matrices[0], pauli_matrices[i+1])

        density_matrix += T_par[i+1, 0] * \
            np.kron(pauli_matrices[i+1], pauli_matrices[0])

    density_matrix *= 0.25

    if run == 0:
        reference_density = density_matrix
        print(' T-matrix:\n', T, '\n')
    else:
        error_samples.append(density_matrix)

error_samples = np.array(error_samples)
mean_density = np.mean(error_samples, axis=0)
# The errors for real and imaginary parts
# correspond to different coefficients and should
# be computed separately
re = np.real(error_samples)
im = np.imag(error_samples)
std_density = np.std(error_samples, axis=0, ddof=1)
std_re_density = np.std(re, axis=0, ddof=1)
std_im_density = np.std(im, axis=0, ddof=1)

np.set_printoptions(precision=5, linewidth=1000, suppress=True)

print(' Theoretical density matrix:\n', reference_density, '\n')
print(' Mean density matrix:\n', mean_density, '\n')
print(' SD of the density matrix:\n',
      std_re_density + 1j * std_im_density, '\n')
print(' Regular SD:\n', std_density, '\n')
