import numpy as np
from numpy import linalg
from scipy.linalg import hilbert
from solver import Solver

def main():

	n_list = [2,4,6,8,10]

	for n in n_list:

		print("\n\n"+"="*100+"\n\nn = ",n,"\n")

		H = hilbert(n)
		b = np.ones(n)

		system = Solver(H,b,"H(n)")

		print("\nLU Decomposition:")
		system.LU()

		print("\nCholesky Decomposition:")
		system.cholesky()

		print("\nQR Decomposition:")
		system.QR()



main()