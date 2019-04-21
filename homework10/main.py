import os
import sys
import eigvals as eig
import numpy as np
from numpy import linalg
from scipy.linalg import hilbert, eigvals
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = "serif"


def get_eigvals(A, figname, Wilkinson = False):

	# get eigenvalues
	m = A.shape[0]
	eigvals = np.zeros(m)

	T = eig.tridiag(A)
	Tnew, t = eig.qralg(T, Wilkinson)
	eigvals[m-1] = Tnew[m-1,m-1]

	for k in range(m-1,1,-1):
		Tnew, tk = eig.qralg(Tnew[:k,:k], Wilkinson)
		t = np.append(t,tk)
		eigvals[k-1] = Tnew[k-1,k-1]

	eigvals[0] = Tnew[0,0]

	# plot
	x = np.array([i for i in range(1,t.shape[0]+1)])
	plt.figure(figsize=(8,6))
	plt.xlim(1, t.shape[0])
	plt.semilogy(x,t,linestyle='solid',linewidth=2)
	plt.ylabel(r"$|t_{m,m-1}|$",fontsize=12)
	plt.xlabel(r"Number of QR factorizations",fontsize=12)

	if Wilkinson:
		title = r"QR Algorithm with Wilkinson Shift"
	else:
		title = r"Pure QR Algorithm"
	plt.title(title,fontsize=16)

	# save and open
	plt.savefig(figname+'.png', format='png')
	#os.system('okular '+figname+'.png')
	plt.clf()

	return -np.sort(-eigvals)


def print_eigvals(eigvals, Wilkinson = False):

	m = eigvals.shape[0]

	if Wilkinson:
		print("\nEigenvalues (QR with Wilkinson shift):")

	else:
		print("\nEigenvalues (Pure QR):")

	for i in range(m):
		print("{:<5d}{:15.10f}".format(i+1,eigvals[i]))



def main():

	np.set_printoptions(precision=4)

	A = hilbert(4)
	print("\nA = \n",A)

	# without Wilkinson shift
	eigvals = get_eigvals(A,'hilbert4')
	print_eigvals(eigvals)

	# with Wilkinson shift
	eigvals = get_eigvals(A,'hilbert4_Wilkinson',True)
	print_eigvals(eigvals,True)



	v = np.array([i for i in range(15,0,-1)])
	A = np.diag(v) + np.ones((15,15))
	print("\nA = \n",A)

	# without Wilkinson shift
	eigvals = get_eigvals(A,'diagones15')
	print_eigvals(eigvals)

	# with Wilkinson shift
	eigvals = get_eigvals(A,'diagones15_Wilkinson',True)
	print_eigvals(eigvals,True)




main()