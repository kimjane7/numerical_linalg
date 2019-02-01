import numpy as np 
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh, expm

class QR_Decomposition:

	def __init__(self, A):

		self.m, self.n = A.shape
		self.A = A
		self.Q = np.zeros_like(A)
		self.R = np.zeros((self.n,self.n))

		self.GramSchmidt_classical()


	def GramSchmidt_classical(self):

		for j in range(self.n):

			# take jth column of A (does this make copy?)
			print(self.A[:,1])
			vj = self.A[:,j]

			# subtract components parallel to previous (j-1) unit vectors      
			for i in range(j-1):

				self.R[i,j] = np.dot(np.conjugate(self.Q[:,i]),self.A[:,j])
				vj = vj - self.R[i,j]*self.A[:,j]

			self.R[j,j] = np.dot(np.conjugate(vj),vj)
			self.Q[:,j] = vj/self.R[j,j]






	#def GramSchmidt_modified(self):


	#def Householder(self):


	#def compare(self):
