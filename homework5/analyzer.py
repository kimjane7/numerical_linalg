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

	def sample_eigvals(self, m, N):

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

		# store average spectral radius for each batch of samples
		full_rho = [full_randmats[i].rho for i in range(N)]
		part_rho = [part_randmats[i].rho for i in range(N)]
		tri_rho = [tri_randmats[i].rho for i in range(N)]
		full_rho_avg = np.mean(full_rho)
		part_rho_avg = np.mean(part_rho)
		tri_rho_avg = np.mean(tri_rho)

		'''
		# store average spectral radius for each batch of samples
		full_norm = [full_randmats[i].norm for i in range(N)]
		part_norm = [part_randmats[i].norm for i in range(N)]
		tri_norm = [tri_randmats[i].norm for i in range(N)]
		full_norm_avg = np.mean(full_norm)
		part_norm_avg = np.mean(part_norm)
		tri_norm_avg = np.mean(tri_norm)
		'''

		return full_eig_real, full_eig_imag, part_eig_real, part_eig_imag, tri_eig_real, tri_eig_imag, \
			   full_rho_avg, part_rho_avg, tri_rho_avg



	def plot_eigenvalues(self, N):

		M = len(self.m_list)

		# get colors for each m
		colors = cm.rainbow(np.linspace(0.0, 1.0, M))

		# make plot
		w = 3
		plt.figure(figsize=(w*M,w*3))
		xymin, xymax = -1.4, 1.4

		# overlay eigenvalues for various m
		for m in self.m_list:

			# get index of m
			m_idx = self.m_list.index(m)

			# take N samples eigenvalues of mxm full, partially triangular, and triangular random matrices
			eigvals = self.sample_eigvals(m,N)
			avg_rhos = eigvals[-3:]

			# superimpose eigenvalues N samples and spectral radii
			for i in range(3):
				subplot = plt.subplot(3, M, M*i+m_idx+1)
				subplot.set_xticklabels([])
				subplot.set_yticklabels([])
				subplot.set_xlim([xymin,xymax])
				subplot.set_ylim([xymin,xymax])
				subplot.scatter(eigvals[2*i], eigvals[2*i+1], c=colors[m_idx], marker='o', edgecolor='none', alpha=0.5)
				
				rho = plt.Circle((0,0), avg_rhos[i], edgecolor='k', fill=False, linewidth=1.0)
				subplot.add_patch(rho)

		plt.tight_layout()
		plt.show()
			

