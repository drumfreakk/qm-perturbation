#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

mass = 1
hbar = 1

def psi_0_m_V_psi_0_n(m,n, alpha, a):
	first  = (np.cos(np.pi * (m-n)) + 1) / (m-n)**2
	second = (np.cos(np.pi * (m+n)) + 1) / (m+n)**2
	return (alpha * a**3 / (4 * np.pi**2)) * (first - second)

def E_0(n, a):
	return n**2 * np.pi**2 * hbar**2 / (2 * mass * a**2)

def psi_0(n, a, x):
	return np.sqrt(2/a) * np.sin(n*np.pi*x/a)

def V(alpha, a, x):
	return alpha * abs(x-a/2)

def psi_1(n, a, alpha, x, limit):
	#sum(m != n) psi_0_m_V_psi_0_n / (E_0_n - E_0_m)  * psi_0_m
	out = 0
	for m in range(1, limit):
		if m != n:
			out += psi_0_m_V_psi_0_n(m,n, alpha, a) * psi_0(m, a, x) / (E_0(n, a) - E_0(m, a))
	return out

def E_1(n, a, alpha):
	return alpha * a * (1/2 + ((-1)**n - 1) / (n**2 * np.pi**2)) / 2

def E_2(n, a, alpha, limit):
	out = 0
	for m in range(1, limit):
		if m != n:
			out += psi_0_m_V_psi_0_n(m,n,alpha,a)**2 / (E_0(n,a) - E_0(m, a))
	return out


a = 1
alpha = 5000
dx = 0.01

expansion = 100

x = np.array([i*dx for i in range(int(a / dx))])

plt.plot(x, V(alpha, a, x))
plt.plot(x, psi_0(1,a,x))
plt.plot(x, psi_0(2,a,x))
plt.plot(x, psi_0(3,a,x))
plt.plot(x, psi_0(1,a,x) + psi_1(1, a, alpha, x, expansion))
plt.plot(x, psi_0(2,a,x) + psi_1(2, a, alpha, x, expansion))
plt.plot(x, psi_0(3,a,x) + psi_1(3, a, alpha, x, expansion))

plt.show()


dalpha = 0.1
alpharange = np.array([i*dalpha for i in range(int(100000/dalpha))])

plt.plot(alpharange, [E_0(1, a) for i in alpharange], label="E_0")
plt.plot(alpharange, E_0(1, a) + E_1(1, a, alpharange), label="E_1")
plt.plot(alpharange, E_0(1, a) + E_1(1, a, alpharange) + E_2(1, a, alpharange, expansion), label="E_2")

for i in range(2, 10):
	print(E_0(i, a) - E_0(i-1, a))
plt.legend()
plt.show()
