import numpy as np 
from numpy import array, triu, random, linalg
from scipy.linalg import svdvals

class RandomMatrix:

	def __init__(self, m, triangular=False):

		self.m = m

		# construct random matrix
		if triangular: 
			self.A = np.triu(np.random.randn(m,m)/np.sqrt(m))
		else:
			self.A = np.random.randn(m,m)/np.sqrt(m)

		# get eigenvalues, norm, spectral radius, and smallest singular value
		self.eigvals = np.linalg.eigvals(self.A)
		self.norm = np.linalg.norm(self.A)
		self.rho = np.sort(np.absolute(self.eigvals))[-1]
		self.sigma_min = np.sort(svdvals(self.A))[0]