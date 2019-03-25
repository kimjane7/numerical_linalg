import numpy as np 
from numpy import array, dot, outer, diag, reshape, transpose, linalg

class Solver:

	def __init__(self, A, b):

		self.m = A.shape[0]
		self.A = A.astype(complex)
		self.b = b.astype(complex)
		self.tolerance = 1E-30

		self.calc_kappa()
		print("kappa = ", self.kappa)


	def calc_kappa(self):

		self.kappa = np.linalg.norm(self.A)*np.linalg.norm(np.linalg.inv(self.A))


	def LU_nopivot(self):

		L = np.eye(self.m, dtype=complex)
		U = self.A.copy()

		for k in range(self.m-1):

			L[k+1:self.m,k] = U[k+1:self.m,k]/U[k,k]
			U[k+1:self.m,k:self.m] -= np.outer(L[k+1:self.m,k],U[k,k:self.m])

		self.forward_sub(L)
		self.backward_sub(U)

		print(self.A-np.dot(L,U))
		print(np.dot(self.A,self.x)-self.b)



	# solve Ly=b for y
	def forward_sub(self, L):

		self.y = np.zeros(self.m, dtype=complex)

		self.y[0] = self.b[0]/L[0,0]
		for i in range(1,self.m):
			self.y[i] = (self.b[i]-np.dot(L[i,:i-1],self.y[:i-1]))/L[i,i]


	# solve Ux=y for x
	def backward_sub(self, U):

		self.x = np.zeros(self.m, dtype=complex)

		self.x[self.m-1] = self.y[self.m-1]/U[self.m-1,self.m-1]
		for i in range(self.m-2,-1,-1):
			print(i)
			self.x[i] = (self.y[i]-np.dot(U[i,i+1:],self.x[i+1:]))/U[i,i]






				