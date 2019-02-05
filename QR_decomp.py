import numpy as np 
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh, expm

class QR_Decomposition:


	def __init__(self, A):

		self.m, self.n = A.shape
		self.A = A
		self.tolerance = 1E-30

		self.classical_GramSchmidt()
		self.modified_GramSchmidt()
		self.Householder()


	def classical_GramSchmidt(self):

		self.Q = np.zeros((self.m,self.n),dtype=complex)
		self.R = np.zeros((self.n,self.n),dtype=complex)

		for j in range(self.n):

			# take jth column of A
			vj = self.A[:,j].copy()

			# subtract components parallel to previous (j-1) unit vectors      
			for i in range(j-1):

				self.R[i,j] = np.vdot(self.Q[:,i],self.A[:,j])
				vj = vj - self.R[i,j]*self.Q[:,i]

			# normalization
			self.R[j,j] = np.vdot(vj,vj)

			# store unit vectors
			if self.R[j,j] > self.tolerance:
				self.Q[:,j] = vj/self.R[j,j]

		self.display_result()


	def modified_GramSchmidt(self):

		self.Q = np.zeros((self.m,self.n),dtype=complex)
		self.R = np.zeros((self.n,self.n),dtype=complex)
		V = self.A.copy()

		for i in range(self.n):

			self.R[i,i] = np.vdot(V[:,i],V[:,i])

			if self.R[i,i] > self.tolerance:
				self.Q[:,i] = V[:,i]/self.R[i,i]

			for j in range(i+1, self.n):
				self.R[i,j] = np.vdot(V[:,i],self.Q[:,j])
				V[:,j] = V[:,j]-self.R[i,j]*self.Q[:,i]

		self.display_result()


	def Householder(self):

		V = np.zeros((self.m,self.n),dtype=complex)

		for k in range(self.n):
			X = self.A[k:self.m,k]


		


	def display_result(self):

		names = ["A","Q","R","A-QR"]
		matrices = [self.A,self.Q,self.R,self.A-np.dot(self.Q,self.R)]
		dims = [self.m,self.m,self.n,self.m]
		widths = [6,6,6,9]

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>5.2f}{:<+5.2f}i'.format(x.real,x.imag) if x.imag > 0.0 else '{:^11.2f}'.format(x.real))})

		for n in range(len(names)):
			print("\n", names[n]," = ",matrices[n][0,:])
			for i in range(1,dims[n]):
				print(" "*widths[n],matrices[n][i,:])

		print("")
