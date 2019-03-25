import numpy as np
from numpy import linalg
from scipy.linalg import hilbert
from solver import Solver

def main():

	n_list = [2,4]

	for n in n_list:

		H = hilbert(n)
		b = np.ones(n)

		system = Solver(H,b)
		system.LU_nopivot()







main()