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
		self.plot_eigenvalues(100)

	def random_samples(self, m, N):

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

		# store average norm for each batch of samples
		full_norm = [full_randmats[i].norm for i in range(N)]
		part_norm = [part_randmats[i].norm for i in range(N)]
		tri_norm = [tri_randmats[i].norm for i in range(N)]
		full_norm_avg = np.mean(full_norm)
		full_norm_std = np.std(full_norm)
		part_norm_avg = np.mean(part_norm)
		part_norm_std = np.std(part_norm)
		tri_norm_avg = np.mean(tri_norm)
		tri_norm_std = np.std(tri_norm)

		# store average spectral radius for each batch of samples
		full_rho = [full_randmats[i].rho for i in range(N)]
		part_rho = [part_randmats[i].rho for i in range(N)]
		tri_rho = [tri_randmats[i].rho for i in range(N)]
		full_rho_avg = np.mean(full_rho)
		full_rho_std = np.std(full_rho)
		part_rho_avg = np.mean(part_rho)
		part_rho_std = np.std(part_rho)
		tri_rho_avg = np.mean(tri_rho)
		tri_rho_std = np.std(tri_rho)

		# store average sigma_min for each batch of samples
		full_sigma_min = [full_randmats[i].sigma_min for i in range(N)]
		part_sigma_min = [part_randmats[i].sigma_min for i in range(N)]
		tri_sigma_min = [tri_randmats[i].sigma_min for i in range(N)]
		full_sigma_min_avg = np.mean(full_sigma_min)
		full_sigma_min_std = np.std(full_sigma_min)
		part_sigma_min_avg = np.mean(part_sigma_min)
		part_sigma_min_std = np.std(part_sigma_min)
		tri_sigma_min_avg = np.mean(tri_sigma_min)
		tri_sigma_min_std = np.std(tri_sigma_min)

		print("{:>4s}{:<3d}".format("m = ",m))
		print("{:<7s}{:^10s}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}".format(" ","full",full_rho_avg,full_rho_std,full_norm_avg,full_norm_std,full_sigma_min_avg,full_sigma_min_std))
		print("{:<7s}{:^10s}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}".format(" ","part",part_rho_avg,part_rho_std,part_norm_avg,part_norm_std,part_sigma_min_avg,part_sigma_min_std))
		print("{:<7s}{:^10s}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}".format(" ","tri",tri_rho_avg,tri_rho_std,tri_norm_avg,tri_norm_std,tri_sigma_min_avg,tri_sigma_min_std))

		return full_eig_real, full_eig_imag, part_eig_real, part_eig_imag, tri_eig_real, tri_eig_imag, \
			   full_rho_avg, part_rho_avg, tri_rho_avg



	def plot_eigenvalues(self, N):

		M = len(self.m_list)
		print("\n{:<7s}{:^10s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(" ","matrix","avg rho","std rho","avg norm","std norm","avg sigma_min","std sigma_min"))

		# get colors for each m
		colors = cm.rainbow(np.linspace(0.0, 1.0, M))

		# make plot
		fig, ax = plt.subplots(3, M, sharex=True, sharey=True)
		xymin, xymax = -1.3, 1.3

		# overlay eigenvalues for various m
		for m in self.m_list:

			# get index of m
			j = self.m_list.index(m)

			# take N samples eigenvalues of mxm full, partially triangular, and triangular random matrices
			data = self.random_samples(m, N)
			avg_rhos = data[-3:]

			# superimpose eigenvalues N samples and spectral radii
			for i in range(3):

				ax[i,j].tick_params(axis='both', labelsize=7)
				ax[i,j].set_xlim([xymin,xymax])
				ax[i,j].set_ylim([xymin,xymax])
				ax[i,j].scatter(data[2*i], data[2*i+1], c=colors[j], marker='o', edgecolor='none', alpha=0.5)

				rho = plt.Circle((0,0), avg_rhos[i], edgecolor='k', fill=False, linewidth=1.0)
				ax[i,j].add_patch(rho)
				ax[i,j].set_aspect('equal')


		plt.subplots_adjust(hspace=0.2, wspace=0.1)
		figname = 'eigenvalues_N'+str(N)+'.png'
		plt.savefig(figname, format='png')
		os.system('okular '+figname)
		plt.clf()
			

