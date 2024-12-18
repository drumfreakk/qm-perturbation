#!/usr/bin/python3

# Code to calculate the energies of the ground state and 1st excited state of
# a particle in an infinite square well, with a sawtooth perturbation potential
# V = alpha * | x - a/2 |
# The energies are calculated for a range of values of alpha,
# and with perturbation theory and exact diagonalisation.
#
# Written for the 2024/25 QM course poster project


import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
from numpy import linalg as LA

### CONFIGURATION & CONSTANTS

# Use Hartree atomic units
mass = 1
hbar = 1

# Set the box width to 1
# The box ranges from x=0 to x=a=1
a = 1

# Define the range to calculate alpha over
max_alpha = 101
dalpha = 1
alpharange = np.array([i*dalpha for i in range(int(max_alpha/dalpha))])


## PERTURBATION THEORY

# Show the contribution of each term of the sum of
# the 2nd order correction in PT for this alpha
display_alpha = 100

# Use this many terms of the sum in the 2nd order PT correction
sum_limit = 15


## EXACT DIAGONALISATION

# The number of gridpoints 
gridpoints = 201


## MATPLOTLIB SETUP

mpl.style.use("classic")


### CODE

## PERTURBATION THEORY

# Returns < psi^0_m | V | psi^0_n >, for a given m, n, alpha and a
def psi_0_m_V_psi_0_n(m,n, alpha, a):
	first  = (np.cos(np.pi * (m-n)) + 1) / ((m-n)**2)
	second = (np.cos(np.pi * (m+n)) + 1) / ((m+n)**2)
	return (alpha * a / (np.pi**2)) * (first - second)

# Returns the uncorrected energy
def E_0(n, a):
	return n**2 * np.pi**2 * hbar**2 / (2 * mass * a**2)

# Returns the 1st order PT correction of the energy
def E_1(n, a, alpha):
	return alpha * a * (1/2 + ((-1)**n - 1) / (n**2 * np.pi**2)) / 2

# Returns the 2nd order PT correction of the energy
# Stops the sum at limit
def E_2(n, a, alpha, limit):

	# Sum from m=1 to m=limit, to calculate the second order correction 
	out = 0
	for m in range(1, limit):
		if m != n:
			out += psi_0_m_V_psi_0_n(m,n,alpha,a)**2 / (E_0(n,a) - E_0(m, a))
	return out

# Return the energy of the nth energy state up to the 2nd order PT correction
def energy_PT(n, a, alpha, limit):
	return E_0(n, a) + E_1(n, a, alpha) + E_2(n, a, alpha, limit)


# To make the energies dimensionless
level_spacing =  E_0(2,a) - E_0(1,a)


# Plot the 2nd order PT energy correction, stopping the sum at different points
# We use this to justify stopping the sum early.
second_order_correction = []
for i in range(2, 21):
	second_order_correction.append(E_2(1,a,display_alpha,i) / level_spacing)

plt.figure(figsize=(17.5,12), facecolor='w')

plt.plot(range(2,21), second_order_correction, 'o-', linewidth=3, markersize=20)

plt.tick_params(labelsize=30)
plt.xlabel("Number of sum terms", fontsize=30)
plt.ylabel("Dimensionless $2^\\mathregular{nd}$ order PT energy correction",
           fontsize=30)

plt.savefig("2o_pt.png", dpi=600)
plt.show()



## EXACT DIAGONALISATION

# Calculate the energy of the first two energy levels for a given alpha,
# using the exact diagonalisation method
def energy_ED(alpha, gridpoints): 
	
	# The spacing between the gridpoints
	gridspacing = a/(gridpoints+1)
	
	#Initialize the Hamiltonian to 0
	H = np.zeros((gridpoints,gridpoints))

	# Already assumes Hartree atomic units
	t_neighbor=1.0/(2.0*gridspacing**2)

	# Populate the Hamiltonian without using a nested loop
	for i in range(gridpoints):
		H[i,i] = 2*t_neighbor + alpha * gridspacing * abs(i - (gridpoints-1)/2)

		# Populate the neigbours
		if i-1 >= 0:
			H[i-1,i] = -t_neighbor
			H[i,i-1] = -t_neighbor
	
	# eigh gets us the (real) eigenvalues and eigenvectors 
	# of a hermitian or real-symmetric matrix.
	unsorted_eigenvalues, unsorted_eigenvectors = LA.eigh(H)

	# the array sortorder holds the indices 
	# to put the eigenvalues in ascending order
	sortorder = np.argsort(unsorted_eigenvalues)
	sorted_eigenvalues = unsorted_eigenvalues[sortorder]
	
	# Only return the energies of the ground state and 1st excited state
	return sorted_eigenvalues[:2]


# Calculate the eigenenergies using exact diagonalisation
# using a range of alpha values
discretized_energies = [[], []]

# Optionally calculate the energies using a larger grid to confirm the results
#discretized_energies_largergrid = [[], []]

for alpha in alpharange:
	energies = energy_ED(alpha, gridpoints)

	# Save the (dimensionless) energies
	discretized_energies[0].append(energies[0]/level_spacing)
	discretized_energies[1].append(energies[1]/level_spacing)
	
#	energies = energy_ED(alpha, 2*(gridpoints-1) + 1)

#	discretized_energies_largergrid[0].append(energies[0]/level_spacing)
#	discretized_energies_largergrid[1].append(energies[1]/level_spacing)


# Output the results into a plot

plt.figure(figsize=(11,18), facecolor='w')

plt.plot(alpharange, discretized_energies[0],
         "orange", label="$E_0$ ED", linewidth=3)
plt.plot(alpharange, energy_PT(1, a, alpharange, sum_limit) / level_spacing,
         "r", label="$E_0$ PT", linewidth=3)
plt.plot(alpharange, discretized_energies[1],
         "b", label="$E_1$ ED", linewidth=3)
plt.plot(alpharange, energy_PT(2, a, alpharange, sum_limit) / level_spacing,
         "g", label="$E_1$ PT", linewidth=3)

#plt.plot(alpharange, discretized_energies_largergrid[0],
#         label="E0 ED large grid", linewidth=3)
#plt.plot(alpharange, discretized_energies_largergrid[1],
#         label="E1 ED large grid", linewidth=3)

plt.tick_params(labelsize=30)
plt.xlabel('$\\alpha$', fontsize=40)
plt.ylabel('Dimensionless energy ($E/(E_1 - E_0)$)', fontsize=30)
plt.legend(fontsize=30, loc="upper left")
plt.savefig("energy.png", dpi=600)
plt.show()
