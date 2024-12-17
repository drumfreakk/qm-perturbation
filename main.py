#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import linalg as LA

# Use Hartree atomic units
mass = 1
hbar = 1

# Set the box width to 1
a = 1

# Define the range to calculate alpha over
max_alpha = 100
dalpha = 0.1
alpharange = np.array([i*dalpha for i in range(int(max_alpha/dalpha))])

# Show the contribution of each term of the sum of the 2nd order correction in PT for this alpha
alpha = 100

# Use this many terms of the sum in the 2nd order PT correction
expansion = 15


### PERTURBATION THEORY

# Returns < psi^0_m | V | psi^0_n >, for a given m, n, alpha and a
def psi_0_m_V_psi_0_n(m,n, alpha, a):
	first  = (np.cos(np.pi * (m-n)) + 1) / ((m-n)**2)
	second = (np.cos(np.pi * (m+n)) + 1) / ((m+n)**2)
	return (alpha * a / (np.pi**2)) * (first - second)

# Returns the perturbation potential
def V(alpha, a, x):
	return alpha * abs(x-a/2)

# Returns the uncorrected wavefunction
def psi_0(n, a, x):
	return np.sqrt(2/a) * np.sin(n*np.pi*x/a)

# Returns the 1st-order PT correction of the wave function
# Stops the sum at limit
def psi_1(n, a, alpha, x, limit):
	#sum(m != n) psi_0_m_V_psi_0_n / (E_0_n - E_0_m)  * psi_0_m
	out = 0
	for m in range(1, limit):
		if m != n:
			out += psi_0_m_V_psi_0_n(m,n, alpha, a) * psi_0(m, a, x) / (E_0(n, a) - E_0(m, a))
	return out

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


# Plot the PT corrections of the ground state
corrs1 = []
corrs2 = []
for i in range(0, 200):
	corrs1.append(E_1(1, a, i) + E_2(1, a, i, expansion))
	corrs2.append(E_1(2, a, i) + E_2(2, a, i, expansion))
plt.plot(range(0,200), corrs1, label="Ground state correction")
plt.plot(range(0,200), corrs2, label="1st excited state correction")
plt.legend()
plt.xlabel("$\\alpha$")
plt.ylabel("Correction energy (a.u.)")
plt.show()


# Plot the 2nd order PT energy correction, stopping the sum at different points
# We use this to justify stopping the sum early.
diff = []
for i in range(4, 21):
	diff.append(E_2(1,a,alpha,i))
plt.plot(range(4,21), diff, 'x-')
plt.xlabel("Number of sum terms")
plt.ylabel("Second-order PT energy correction (a.u.)")
plt.show()


## EXACT DIAGONALISATION
# Constants
N_gridpoints = 203                             # number of ghrid points
L_Box_size = 1#20                                 # box size (in units of a0)
a_gridspacing= L_Box_size/(N_gridpoints+1)      # grid spacing (in units of a0)

#Building the hamiltonian matrix
Eig_Energies1=[]
Eig_Energies2=[]
H = np.zeros((N_gridpoints,N_gridpoints))   # initialize the Hamiltonian matrix to zero
t_neighbor=1.0/(2.0*a_gridspacing**2)
def Calc_Eigenenergies(alpha): # the coupling between neighboring elements
    for i in range (0,N_gridpoints) :
       for j in range (0,N_gridpoints) :
          if i==j : H[i,j] = 2*t_neighbor+alpha*a_gridspacing*abs((i)-math.ceil((N_gridpoints-1)/2))          # set the diagonal elements
          if abs(i-j) == 1 : H[i,j] = -t_neighbor # set the side-diagonal elements
    unsorted_eigenvalues, unsorted_eigenvectors = LA.eigh(H) # eigh gets us the (real) eigenvalues and eigenvectors of a hermitian or real-symmetric matrix.
    sortorder = np.argsort(unsorted_eigenvalues)             # the array sortorder holds the indices to put the eigenvalues in ascending order
    sorted_eigenvectors = np.transpose(unsorted_eigenvectors)[sortorder] #Transpose is needed because of the way eig and eigh return the eigenvectors.
    sorted_eigenvalues = unsorted_eigenvalues[sortorder]
    return sorted_eigenvalues[0:2]


for x in range(0,max_alpha):
    Eig_Energies1.append(Calc_Eigenenergies(x)[0])
    Eig_Energies2.append(Calc_Eigenenergies(x)[1])

   #end for j
#end for i
# Note that this way of filling the matrix using a nested for-loop is rather slow and thewre are more efficient ways.


# Output the results into plots

n_range=range(0,len(Eig_Energies1))

#E_theory_range=n_range**2*np.pi**2 /(2* L_Box_size**2) # Energies according to Griffiths 2.30;

plt.figure(figsize=(16,12))
plt.plot(n_range,Eig_Energies1)
plt.plot(n_range, Eig_Energies2)
plt.xlabel('$\\alpha$', fontsize=20)
plt.ylabel('Energy (a.u.)', fontsize=20)

plt.plot(alpharange, E_0(1, a) + E_1(1, a, alpharange) + E_2(1, a, alpharange, expansion), label="Ground state PT")
plt.plot(alpharange, E_0(2, a) + E_1(2, a, alpharange) + E_2(2, a, alpharange, expansion), label="1st excited PT")

plt.legend(['Ground State','First Exited State', "PT 1", "PT 2"], fontsize=25)
plt.show()
