import os
import sys
import numpy as np 
from numpy import array, dot, diag
from randmat import RandomMatrix

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.family'] = "serif"


class Analyzer:

	def __init__(self, m_list, sigma_min_bounds):

		self.m_list = m_list
		self.bounds = sigma_min_bounds
		#self.plot_eigenvalues(100)
		#self.plot_norm_rho()
		self.print_sigma_min()

	def random_samples(self, N, m):

		# construct lists of N random matrices with zeros below kth subdiagonal
		full_randmats = [RandomMatrix(m,-m+1) for i in range(N)]
		hess_randmats = [RandomMatrix(m,-1) for i in range(N)]
		tri_randmats = [RandomMatrix(m,0) for i in range(N)]

		# construct matrices of eigenvalues
		full_eig_real = np.zeros((N,m))
		full_eig_imag = np.zeros((N,m))
		hess_eig_real = np.zeros((N,m))
		hess_eig_imag = np.zeros((N,m))
		tri_eig_real = np.zeros((N,m))
		tri_eig_imag = np.zeros((N,m))

		for i in range(N):
			full_eig_real[i] = full_randmats[i].eigvals.real
			full_eig_imag[i] = full_randmats[i].eigvals.imag
			hess_eig_real[i] = hess_randmats[i].eigvals.real
			hess_eig_imag[i] = hess_randmats[i].eigvals.imag
			tri_eig_real[i] = tri_randmats[i].eigvals.real
			tri_eig_imag[i] = tri_randmats[i].eigvals.imag

		# reshape into vectors
		full_eig_real.reshape(-1)
		full_eig_imag.reshape(-1)
		hess_eig_real.reshape(-1)
		hess_eig_imag.reshape(-1)
		tri_eig_real.reshape(-1)
		tri_eig_imag.reshape(-1)

		# store average norm for each batch of samples
		full_norm = [full_randmats[i].norm for i in range(N)]
		hess_norm = [hess_randmats[i].norm for i in range(N)]
		tri_norm = [tri_randmats[i].norm for i in range(N)]
		full_norm_avg = np.mean(full_norm)
		full_norm_std = np.std(full_norm)
		hess_norm_avg = np.mean(hess_norm)
		hess_norm_std = np.std(hess_norm)
		tri_norm_avg = np.mean(tri_norm)
		tri_norm_std = np.std(tri_norm)

		# store average spectral radius for each batch of samples
		full_rho = [full_randmats[i].rho for i in range(N)]
		hess_rho = [hess_randmats[i].rho for i in range(N)]
		tri_rho = [tri_randmats[i].rho for i in range(N)]
		full_rho_avg = np.mean(full_rho)
		full_rho_std = np.std(full_rho)
		hess_rho_avg = np.mean(hess_rho)
		hess_rho_std = np.std(hess_rho)
		tri_rho_avg = np.mean(tri_rho)
		tri_rho_std = np.std(tri_rho)

		# store average sigma_min for each batch of samples
		full_sigma_min = [full_randmats[i].sigma_min for i in range(N)]
		hess_sigma_min = [hess_randmats[i].sigma_min for i in range(N)]
		tri_sigma_min = [tri_randmats[i].sigma_min for i in range(N)]
		full_sigma_min_avg = np.mean(full_sigma_min)
		full_sigma_min_std = np.std(full_sigma_min)
		hess_sigma_min_avg = np.mean(hess_sigma_min)
		hess_sigma_min_std = np.std(hess_sigma_min)
		tri_sigma_min_avg = np.mean(tri_sigma_min)
		tri_sigma_min_std = np.std(tri_sigma_min)

		# get proportions of matrices with sigma_min <= sigma_min_bound
		ratios = np.zeros((3,len(self.bounds)))
		for bound in self.bounds:

			if bound > 0.0:

				j = self.bounds.index(bound)

				full_ratio = 0.0
				hess_ratio = 0.0
				tri_ratio = 0.0

				for i in range(N):

					if full_randmats[i].sigma_min <= bound:
						ratios[0,j] += 1.0
					if hess_randmats[i].sigma_min <= bound:
						ratios[1,j] += 1.0
					if tri_randmats[i].sigma_min <= bound:
						ratios[2,j] += 1.0

				ratios[0,j] = ratios[0,j]/N
				ratios[1,j] = ratios[1,j]/N
				ratios[2,j] = ratios[2,j]/N

		'''
		print("{:>4s}{:<3d}".format("m = ",m))
		print("{:<7s}{:^10s}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}".format(" ","full",full_rho_avg,full_rho_std,full_norm_avg,full_norm_std,full_sigma_min_avg,full_sigma_min_std))
		print("{:<7s}{:^10s}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}".format(" ","hess",hess_rho_avg,hess_rho_std,hess_norm_avg,hess_norm_std,hess_sigma_min_avg,hess_sigma_min_std))
		print("{:<7s}{:^10s}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}{:^15.7f}".format(" ","tri",tri_rho_avg,tri_rho_std,tri_norm_avg,tri_norm_std,tri_sigma_min_avg,tri_sigma_min_std))
		'''

		return full_eig_real, full_eig_imag, hess_eig_real, hess_eig_imag, tri_eig_real, tri_eig_imag, \
			   full_norm_avg, hess_norm_avg, tri_norm_avg, \
			   full_rho_avg, hess_rho_avg, tri_rho_avg, \
			   full_sigma_min_avg, hess_sigma_min_avg, tri_sigma_min_avg,\
			   full_norm_std, hess_norm_std, tri_norm_std, \
			   full_rho_std, hess_rho_std, tri_rho_std, \
			   full_sigma_min_std, hess_sigma_min_std, tri_sigma_min_std, \
			   ratios


	def plot_eigenvalues(self, N):

		M = len(self.m_list)
		#print("\n{:<7s}{:^10s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(" ","matrix","avg rho","std rho","avg norm","std norm","avg sigma_min","std sigma_min"))

		# get colors for each m
		colors = cm.rainbow(np.linspace(0.0, 1.0, M))

		# make plot
		fig, ax = plt.subplots(3, M, sharex=True, sharey=True)
		matplotlib.rcParams['axes.unicode_minus'] = False
		xlabels = [' ',-1,' ',0,' ',1,' ']
		ylabels = [' ','-i',' ',0,' ','i',' ']
		xymin, xymax = -1.5, 1.5

		# overlay eigenvalues for various m
		for m in self.m_list:

			# get index of m
			j = self.m_list.index(m)

			# take N samples eigenvalues of mxm full, hessially triangular, and triangular random matrices
			data = self.random_samples(N, m)
			avg_rhos = data[9:12]

			# title
			ax[0,j].set_title('m = '+str(m),fontsize=12)

			# superimpose eigenvalues N samples and spectral radii
			for i in range(3):

				ax[i,j].set_xlim([xymin,xymax])
				ax[i,j].set_ylim([xymin,xymax])
				ax[i,j].set_xticklabels(xlabels)
				ax[i,j].set_yticklabels(ylabels)
				ax[i,j].tick_params(axis='both', labelsize=9, direction='inout', length=4.0)
				ax[i,j].scatter(data[2*i], data[2*i+1], c=colors[j], marker='o', s=2.0, edgecolor='none', alpha=0.7)

				rho = plt.Circle((0,0), avg_rhos[i], edgecolor='k', fill=False, linewidth=0.7)
				ax[i,j].add_patch(rho)
				ax[i,j].text(1.35,-1.35,r'$\overline{\rho}=$'+"{:.3f}".format(avg_rhos[i]), fontsize=9, ha='right')
				ax[i,j].set_aspect('equal')
				ax[i,j].set_adjustable('box-forced')

		ax[0,0].set_ylabel('full',fontsize=12)
		ax[1,0].set_ylabel('upper-Hessenberg',fontsize=12)
		ax[2,0].set_ylabel('upper-triangular',fontsize=12)
		fig.suptitle('Eigenvalues of '+str(N)+' Random Matrices', fontsize=16, va='top')
		plt.tight_layout(rect=[0, 0.03, 1, 0.95])

		# save and open
		figname = 'eigenvalues_N'+str(N)+'.png'
		plt.savefig(figname, format='png')
		os.system('okular '+figname)
		plt.clf()
			


	def plot_norm_rho(self):

		# define a finer mesh of m
		m_list = [i for i in range(self.m_list[0],self.m_list[-1]+2,2)]
		M = len(m_list)
		N = 200
		#print("\n{:<7s}{:^10s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(" ","matrix","avg rho","std rho","avg norm","std norm","avg sigma_min","std sigma_min"))

		# store stats
		avg_norms = np.zeros((3,M))
		avg_rhos = np.zeros((3,M))
		std_norms = np.zeros((3,M))
		std_rhos = np.zeros((3,M))

		for m in m_list:

			# get index of m
			j = m_list.index(m)

			# take many samples
			data = self.random_samples(N, m)

			# extract stats
			avg_norms[:,j] = data[6:9]
			avg_rhos[:,j] = data[9:12]
			std_norms[:,j] = data[15:18]
			std_rhos[:,j] = data[18:21]


		# plot stats
		plt.figure(figsize=(8,6))
		plt.xlim([m_list[0],m_list[-1]])

		plt.plot(m_list, avg_norms[0], c='b', label=r'full $\overline{\| A \|}_2$', lw=1.5)
		plt.fill_between(m_list, avg_norms[0]-std_norms[0], avg_norms[0]+std_norms[0], facecolor='b', alpha=0.3)
		plt.plot(m_list, avg_norms[1], c='r', label=r'Hessenberg $\overline{\| A \|}_2$', lw=1.5)
		plt.fill_between(m_list, avg_norms[1]-std_norms[1], avg_norms[1]+std_norms[1], facecolor='r', alpha=0.3)
		plt.plot(m_list, avg_norms[2], c='g', label=r'triangular $\overline{\| A \|}_2$', lw=1.5)
		plt.fill_between(m_list, avg_norms[2]-std_norms[2], avg_norms[2]+std_norms[2], facecolor='g', alpha=0.3)

		plt.plot(m_list, avg_rhos[0], c='b', linestyle='dashed', label=r'full $\overline{\rho(A)}$', lw=1.5)
		plt.fill_between(m_list, avg_rhos[0]-std_rhos[0], avg_rhos[0]+std_rhos[0], facecolor='b', alpha=0.3)
		plt.plot(m_list, avg_rhos[1], c='r', linestyle='dashed', label=r'Hessenberg $\overline{\rho(A)}$', lw=1.5)
		plt.fill_between(m_list, avg_rhos[1]-std_rhos[1], avg_rhos[1]+std_rhos[1], facecolor='r', alpha=0.3)
		plt.plot(m_list, avg_rhos[2], c='g', linestyle='dashed', label=r'triangular $\overline{\rho(A)}$', lw=1.5)
		plt.fill_between(m_list, avg_rhos[2]-std_rhos[2], avg_rhos[2]+std_rhos[2], facecolor='g', alpha=0.3)

		plt.xlabel(r'dimension $m$',fontsize=12)
		plt.legend(loc='upper left', shadow=True, fontsize=12)
		plt.title(r'Norms and Spectral Radii of '+str(N)+' Random Matrices')

		# save and open
		figname = 'norm_rho_N'+str(N)+'.png'
		plt.savefig(figname, format='png')
		os.system('okular '+figname)
		plt.clf()


	def print_sigma_min(self):

		# define a finer mesh of m
		M = len(self.m_list)
		B = len(self.bounds)
		N = 100
		#print("\n{:<7s}{:^10s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(" ","matrix","avg rho","std rho","avg norm","std norm","avg sigma_min","std sigma_min"))

		avg_sigma_min = np.zeros((3,M))
		std_sigma_min = np.zeros((3,M))
		ratios = np.zeros((M,3,B))

		for m in self.m_list:

			# get index of m
			j = self.m_list.index(m)

			# extract stats
			data = self.random_samples(N,m)

			avg_sigma_min[:,j] = data[12:15]
			std_sigma_min[:,j] = data[21:24]
			ratios[j,:,:] = data[-1]

		# print
		print("Proportions of matrices with sigma_min <= bound")
		for i in range(M):

			print("{:>4s}{:<3d}".format("m = ",self.m_list[i]))

			for k in range(B): 

				print("{:>7s}{:<7s}{:<7.3f}".format(" ","bound = ",self.bounds[k]))

				print("{:<21s}{:>5s}{:>14.5f}".format(" ","full",ratios[i,0,k]))
				print("{:<21s}{:>5s}{:>14.5f}".format(" ","hess",ratios[i,1,k]))
				print("{:<21s}{:>5s}{:>14.5f}".format(" ","tri",ratios[i,2,k]))

		