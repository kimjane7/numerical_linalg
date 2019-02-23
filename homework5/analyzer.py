import os
import sys
import numpy as np 
from numpy import array, dot, diag
from randmat import RandomMatrix

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Analyzer:

	def __init__(self, m_list, sigma_min_bounds):

		self.m_list = m_list
		self.bounds = sigma_min_bounds
		self.plot_eigenvalues(10)

	def plot_eigenvalues(self, N):

		# get colors for each m
		colors = cm.rainbow(np.linspace(0.0,1.0,len(self.m_list)))

		# overlay eigenvalues for various m
		for m in self.m_list:

			# make plot
			plt.figure(figsize=(12,6))

			# construct lists of N random matrices with elements
			# below kth lower diagonal is zero
			full_randmats = [RandomMatrix(m,-m+1) for i in range(N)]
			part_randmats = [RandomMatrix(m,-1) for i in range(N)]
			tri_randmats = [RandomMatrix(m,0) for i in range(N)]

			# construct matrices of eigenvalues
			full_eig_real = np.zeros((N,m))
			full_eig_imag = np.zeros((N,m))
			part_eig_real = np.zeros((N,m))
			part_eig_imag = np.zeros((N,m))
			tri_eig_real = np.zeros((N,m))
			tri_eig_imag = np.zeros((N,m))

			for i in range(N):
				full_eig_real[i] = full_randmats[i].eigvals.real
				full_eig_imag[i] = full_randmats[i].eigvals.imag
				part_eig_real[i] = part_randmats[i].eigvals.real
				part_eig_imag[i] = part_randmats[i].eigvals.imag
				tri_eig_real[i] = tri_randmats[i].eigvals.real
				tri_eig_imag[i] = tri_randmats[i].eigvals.imag

			# reshape into vectors
			full_eig_real.reshape(-1)
			full_eig_imag.reshape(-1)
			part_eig_real.reshape(-1)
			part_eig_imag.reshape(-1)
			tri_eig_real.reshape(-1)
			tri_eig_imag.reshape(-1)

			
			# superimpose eigenvalues of full random matrices
			plt.subplot(1, 3, 1)
			plt.scatter(full_eig_real, full_eig_imag, c=colors[self.m_list.index(m)], marker='o', edgecolor='none', alpha=0.5)

			# superimpose eigenvalues of partially triangular random matrices
			plt.subplot(1, 3, 2)
			plt.scatter(part_eig_real, part_eig_imag, c=colors[self.m_list.index(m)], marker='o', edgecolor='none', alpha=0.5)

			# superimpose eigenvalues of triangular random matrices
			plt.subplot(1, 3, 3)
			plt.scatter(tri_eig_real, tri_eig_imag, c=colors[self.m_list.index(m)], marker='o', edgecolor='none', alpha=0.5)


			

			plt.show()
			

