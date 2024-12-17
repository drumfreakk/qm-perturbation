#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import linalg as LA

### CONFIGURATION & CONSTANTS

# Use Hartree atomic units
mass = 1
hbar = 1

# Set the box width to 1
a = 1

# Define the range to calculate alpha over
max_alpha = 100
dalpha = 1
alpharange = np.array([i*dalpha for i in range(int(max_alpha/dalpha))])


## PERTURBATION THEORY

# Show the contribution of each term of the sum of the 2nd order correction in PT for this alpha
display_alpha = 100

# Use this many terms of the sum in the 2nd order PT correction
sum_limit = 15


## EXACT DIAGONALISATION

# The number of gridpoints 
gridpoints = 201

# The spacing between the gridpoints
gridspacing= a/(gridpoints+1) # TODO: fucked up why this is +


### CODE

## PERTURBATION THEORY

# Returns < psi^0_m | V | psi^0_n >, for a given m, n, alpha and a
def psi_0_m_V_psi_0_n(m,n, alpha, a):
	first  = (np.cos(np.pi * (m-n)) + 1) / ((m-n)**2)
	second = (np.cos(np.pi * (m+n)) + 1) / ((m+n)**2)
	return (alpha * a / (np.pi**2)) * (first - second)

## Returns the perturbation potential
#def V(alpha, a, x):
#	return alpha * abs(x-a/2)
#
## Returns the uncorrected wavefunction
#def psi_0(n, a, x):
#	return np.sqrt(2/a) * np.sin(n*np.pi*x/a)
#
## Returns the 1st-order PT correction of the wave function
## Stops the sum at limit
#def psi_1(n, a, alpha, x, limit):
#	#sum(m != n) psi_0_m_V_psi_0_n / (E_0_n - E_0_m)  * psi_0_m
#	out = 0
#	for m in range(1, limit):
#		if m != n:
#			out += psi_0_m_V_psi_0_n(m,n, alpha, a) * psi_0(m, a, x) / (E_0(n, a) - E_0(m, a))
#	return out

# Returns the uncorrected energy
def E_0(n, a):
	return n**2 * np.pi**2 * hbar**2 / (2 * mass * a**2)

# Returns the 1st order PT correction of the energy
def E_1(n, a, alpha):
	return alpha * a * (1/2 + ((-1)**n - 1) / (n**2 * np.pi**2)) / 2

# Returns the 2nd order PT correction of the energy
# Stops the sum at limit
def E_2(n, a, alpha, limit):
	out = 0
	for m in range(1, limit):
		if m != n:
			out += psi_0_m_V_psi_0_n(m,n,alpha,a)**2 / (E_0(n,a) - E_0(m, a))
	return out


## Plot the PT corrections of the ground state
#corrs1 = []
#corrs2 = []
#for i in range(0, 200):
#	corrs1.append(E_1(1, a, i) + E_2(1, a, i, sum_limit))
#	corrs2.append(E_1(2, a, i) + E_2(2, a, i, sum_limit))
#plt.plot(range(0,200), corrs1, label="Ground state correction")
#plt.plot(range(0,200), corrs2, label="1st excited state correction")
#plt.legend()
#plt.xlabel("$\\alpha$")
#plt.ylabel("Correction energy (a.u.)")
#plt.show()
#
#
## Plot the 2nd order PT energy correction, stopping the sum at different points
## We use this to justify stopping the sum early.
#diff = []
#for i in range(4, 21):
#	diff.append(E_2(1,a,display_alpha,i))
#plt.plot(range(4,21), diff, 'x-')
#plt.xlabel("Number of sum terms")
#plt.ylabel("Second-order PT energy correction (a.u.)")
#plt.show()



## EXACT DIAGONALISATION

# Calculate the energy of the first two energy levels for a given alpha, using the exact diagonalisation method
def calculate_energies(alpha): 
	
	#Initialize the Hamiltonian to 0
	H = np.zeros((gridpoints,gridpoints))

	# Already assumes Hartree atomic units
	t_neighbor=1.0/(2.0*gridspacing**2)

	# Populate the Hamiltonian without using a nested loop
	for i in range(gridpoints):
#TODO
#		print("\nmine", abs(i * gridspacing - a/2))
#		print("orig", gridspacing * abs(i - (gridpoints-1)/2))
#		H[i,i] = 2 * t_neighbor + alpha * int(abs(i * gridspacing - a/2))
		H[i,i] = 2 * t_neighbor + alpha * gridspacing * abs(i - (gridpoints-1)/2)

		# Populate the neigbours
		if i-1 >= 0:
			H[i-1,i] = -t_neighbor
			H[i,i-1] = -t_neighbor
	
#	print(H)

	# eigh gets us the (real) eigenvalues and eigenvectors of a hermitian or real-symmetric matrix.
	unsorted_eigenvalues, unsorted_eigenvectors = LA.eigh(H)

	# the array sortorder holds the indices to put the eigenvalues in ascending order
	sortorder = np.argsort(unsorted_eigenvalues)
	sorted_eigenvalues = unsorted_eigenvalues[sortorder]
	
	# Only return the energies of the ground state and 1st excited state
	return sorted_eigenvalues[:2]


discretized_energies = [[], []]

for alpha in alpharange:
	energies = calculate_energies(alpha)
	discretized_energies[0].append(energies[0])
	discretized_energies[1].append(energies[1])


# Output the results into plots

plt.figure(figsize=(16,12))
plt.plot(alpharange, discretized_energies[0], label="Ground state ED")
plt.plot(alpharange, discretized_energies[1], label="1st excited state ED")

plt.plot(alpharange, E_0(1, a) + E_1(1, a, alpharange) + E_2(1, a, alpharange, sum_limit), label="Ground state PT")
plt.plot(alpharange, E_0(2, a) + E_1(2, a, alpharange) + E_2(2, a, alpharange, sum_limit), label="1st excited state PT")

plt.xlabel('$\\alpha$', fontsize=20)
plt.ylabel('Energy (a.u.)', fontsize=20)
plt.legend(fontsize=25)
plt.show()
