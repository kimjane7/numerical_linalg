import os
import sys
from sturm_liouville import SL_Solver
import numpy as np
from numpy import linalg
from scipy.linalg import expm, eigvals
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.family'] = "serif"


def plot_convergence(N_list, jmax, p, q):

	N = np.array(N_list)
	h = np.pi/N
	lambdah_hat_diff = np.zeros((len(h),jmax))
	lambdah_diff = np.zeros((len(h),jmax))
	schemeA_diff = np.zeros((len(h),jmax))

	for i in range(len(h)):
		
		V0 = np.ones(N_list[i])
		solver = SL_Solver(N_list[i],jmax,p,q)
		solver.schemeA(V0,0.0,100)

		lambdah_hat_diff[i,:] = abs(solver.lambdah_hat-solver.lambda_exact)
		lambdah_diff[i,:] = abs(solver.lambdah-solver.lambda_exact)
		schemeA_diff[i,:] = abs(solver.schemeA_eigvals-solver.lambda_exact)


	plt.figure(figsize=(8,6))

	for j in range(jmax):

		plt.loglog(h,lambdah_hat_diff[:,j],color='red',lw=3)
		plt.loglog(h,lambdah_diff[:,j],color='green')
	
	plt.loglog(h,schemeA_diff[:,0],color='blue')
	

	figname = 'convergence.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()




def main():

	N_list = [2**n for n in range(4,10)]
	plot_convergence(N_list,5,1.0,5.0)

	'''
	solver = SL_Solver(100,5,1.0,5.0)

	V0 = np.ones(100)
	mu = 0.0
	kmax = 1000
	solver.schemeA(V0,mu,kmax)
	'''


main()