import os
import sys
import numpy as np
from numpy import linalg
from scipy.linalg import expm, eigvals
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.family'] = "serif"


def randmat(dim):

	return np.random.randn(dim,dim)-2.0


def norm_exp(t, A):

	norm_exp = np.zeros_like(t)

	for i in range(len(t)):

		norm_exp[i] = np.linalg.norm(expm(t[i]*A))

	return norm_exp


def largest_eigval(A):

	return np.sort(np.linalg.eigvals(A))[-1]


def exp_alpha(t, A):

	alpha = largest_eigval(A).real

	exp_alpha = np.zeros_like(t)

	for i in range(len(t)):

		exp_alpha[i] = np.exp(t[i]*alpha)

	return exp_alpha


def main():

	dim = 4
	n_randmats = 10
	t = np.linspace(0.0, 20.0, 100)
	colors = cm.rainbow(np.linspace(0.0, 1.0, n_randmats))
	np.set_printoptions(precision=4)

	# make figure
	plt.figure(figsize=(8,6))

	# print heading
	print("{0:^20s}{1:^20s}".format("A","largest eigval"))
	print("="*40)

	for n in range(n_randmats):

		# sample random matrices
		A = randmat(dim)

		# format largest eigenvalue
		tolerance = 1E-5
		if(abs(largest_eigval(A).imag) < tolerance):
			label = str(round(largest_eigval(A).real,2))
		else:
			label = str(round(largest_eigval(A).real,2))+"+"+str(round(largest_eigval(A).imag,2))+"i"

		# print largest eigenvalue for each sample
		print("{0:^20s}{1:^20s}".format("("+str(n+1)+")", label))

		# calculate ||exp(tA)|| and exp(t*alpha(A))
		y1 = norm_exp(t, A)
		y2 = exp_alpha(t, A)

		# plot
		plt.semilogy(t,y1,color=colors[n],label=label,linestyle='solid',linewidth=2)
		plt.semilogy(t,y2,color=colors[n],linestyle='dotted',linewidth=2)


	plt.xlabel(r"$t$",fontsize=12)
	plt.legend(loc='upper left', shadow=True, fontsize=12)
	plt.title(r"$\| e^{tA} \|_2$ (solid) and $e^{t\alpha(A)}$ (dotted)")

	# save and open
	figname = 'plot.png'
	plt.savefig(figname, format='png')
	os.system('okular '+figname)
	plt.clf()

main()