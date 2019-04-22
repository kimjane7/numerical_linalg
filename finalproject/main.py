import os
import sys
from sturm_liouville import SL_Solver
import numpy as np
from numpy import linalg
from scipy.linalg import expm, eigvals
from scipy.stats import linregress
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.family'] = "serif"



def plot_convergence(p, q):

	N = np.array([2**n for n in range(4,10)])
	h = np.pi/N
	eigval_diff = np.zeros(len(h))
	x = np.pi/np.array([2**n for n in range(2,12)])

	plt.figure(figsize=(8,6))

	# loop through methods used to compute smallest eigenvalue in scheme A
	method = ['inverse power', 'shifted power', 'QR']
	color = ['blue', 'red', 'green']
	size = [12,9,6]
	linewidth = [3,2,1]
	for index in range(2):

		# loop through step sizes
		for i in range(len(h)):
			
			V0 = np.ones(N[i])
			solver = SL_Solver(N[i],1,p,q)
			mu = [0.0,1000.0,50.0]
			solver.schemeA(V0,method[index],100,mu[index])

			eigval_diff[i] = abs(solver.lambda_exact-solver.lambdah_hat)

		# linear regression
		linreg = linregress(np.log(h),np.log(eigval_diff))
		m = linreg[0]
		b = linreg[1]

		# plot
		plt.loglog(h,eigval_diff,c=color[index],marker='o',markersize=size[index], markeredgewidth=0, lw=0,label=method[index]+' method')
		plt.loglog(x,np.exp(b)*x**m,c=color[index],lw=linewidth[index],label=r'linear regression (y = '+str(round(m,5))+'x'+str(round(b,5))+')')


	plt.title(r'Convergence of smallest eigenvalue in scheme A')
	plt.xlabel(r'$\log(h)$')
	plt.ylabel(r'$\log | \lambda_0-\hat{\lambda}_0^h |$')
	plt.legend(loc='upper center', shadow=True, fontsize=12)		

	figname = 'convergence_schemeA.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()




def main():

	
	plot_convergence(1.0,5.0)


main()