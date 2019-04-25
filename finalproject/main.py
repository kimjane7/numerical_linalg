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

	#N = np.array([4,8])
	N = np.array([2**n for n in range(4,10)])
	h = np.pi/N
	eigval_diff = np.zeros(len(h))
	x = np.pi/np.array([2**n for n in range(2,12)])

	scheme = ['A','A','A','B']
	method = ['inverse power', 'shifted power', 'QR','inverse power']
	nametag = ['inverse', 'shifted', 'QR','inverse']
	color = ['blue', 'red', 'green','blue']
	tolerance = [1E-12,1E-12,1E-12,1E-12]

	for index in range(4):

		print("="*53)
		print("{:^10s}{:^20s}{:^20s}".format('N', 'iterations', 'smallest eigenvalue'))
		print("="*53)

		# loop through step sizes
		for i in range(len(h)):

			solver = SL_Solver(N[i],p,q,tolerance[index])

			if scheme[index] == 'A':
				solver.schemeA(method[index])
				eigval_diff[i] = abs(solver.lambda_exact[0]-solver.schemeA_eigval)
			
			if scheme[index] == 'B':
				solver.schemeB(method[index])
				eigval_diff[i] = abs(solver.lambda_exact[0]-solver.schemeB_eigval)

		# linear regression
		linreg = linregress(np.log(h),np.log(eigval_diff))
		m = linreg[0]
		b = linreg[1]

		'''
		# linear regression excluding outliers
		linreg = linregress(np.log(h)[:-2],np.log(eigval_diff)[:-2])
		n = linreg[0]
		c = linreg[1]
		'''
		
		# plot
		plt.figure(figsize=(8,6))
		plt.loglog(h,eigval_diff,c=color[index],marker='o',markersize=9, markeredgewidth=0, lw=0,label=method[index]+' method (tolerance = '+str(tolerance[index])+')')
		plt.loglog(x,np.exp(b)*x**m,c=color[index],lw=2,label=r'linear regression (y = '+str(round(m,5))+'x'+str(round(b,5))+')')
		#plt.loglog(x,np.exp(c)*x**n,c=color[index],lw=2,linestyle='dashed',label=r'linear regression excluding outliers (y = '+str(round(n,5))+'x'+str(round(c,5))+')')

		plt.title(r'Convergence of smallest eigenvalue (scheme '+scheme[index]+')')
		plt.xlabel(r'$h$')
		plt.ylabel(r'$| \lambda_1-\hat{\lambda}_1^h |$')
		plt.legend(loc='upper left', shadow=True, fontsize=10)		

		# save
		figname = 'convergence_scheme'+scheme[index]+'_'+nametag[index]+'.png'
		plt.savefig(figname, format='png')
		os.system('okular '+figname)
		plt.clf()




def main():

	
	plot_convergence(1.0,5.0)


main()