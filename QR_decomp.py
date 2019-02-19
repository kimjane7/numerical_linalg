import numpy as np 
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh, expm

class QR_Decomposition:

	def __init__(self, A):

		self.m, self.n = A.shape
		self.A = A
		self.tolerance = 1E-30


	def classical_GramSchmidt(self):

		self.Q = np.zeros((self.m,self.n),dtype=complex)
		self.R = np.zeros((self.n,self.n),dtype=complex)

		for j in range(0,self.n):

			# take jth column of A
			vj = self.A[:,j].copy()

			# subtract components parallel to previous (j-1) unit vectors      
			for i in range(0,j):

				self.R[i,j] = np.vdot(self.Q[:,i],self.A[:,j])
				vj = vj-self.R[i,j]*self.Q[:,i]

			# normalization
			self.R[j,j] = np.linalg.norm(vj)

			# store unit vectors
			if self.R[j,j] > self.tolerance:
				self.Q[:,j] = vj/self.R[j,j]


	def modified_GramSchmidt(self):

		self.Q = np.zeros((self.m,self.n),dtype=complex)
		self.R = np.zeros((self.n,self.n),dtype=complex)
		V = self.A.copy()

		for i in range(self.n):

			# normalization
			self.R[i,i] = np.linalg.norm(V[:,i])

			# calculate Q
			if self.R[i,i] > self.tolerance:
				self.Q[:,i] = V[:,i]/self.R[i,i]

			# calculate R
			for j in range(i+1, self.n):
				self.R[i,j] = np.vdot(self.Q[:,i],V[:,j])
				V[:,j] = V[:,j]-self.R[i,j]*self.Q[:,i]



	def Householder(self):

		self.Q = np.zeros((self.m,self.m),dtype=complex)
		self.R = self.A.copy()
		V = np.zeros((self.m,self.n),dtype=complex)

		for k in range(self.n):

			# householder transformation
			x = self.R[k:self.m,k]

			# construct elementary vector
			e1 = np.zeros(self.m-k)
			e1[0] = 1.0

			# get sign
			sgn = np.sign(x[0])
			if sgn == 0:
				sgn = 1+0j	

			# reflection vector
			vk = sgn*np.linalg.norm(x)*e1+x
			if np.linalg.norm(vk) > self.tolerance:
				vk = vk/np.linalg.norm(vk)

			# store to calculate Q
			V[k:self.m,k] = vk

			# calculate R
			self.R[k:self.m,k:self.n] = self.R[k:self.m,k:self.n]-2.0*np.outer(vk,np.dot(np.conjugate(vk),self.R[k:self.m,k:self.n]))

		# calculate Q*I
		for i in range(self.m):

			ei = np.zeros(self.m,dtype=complex)
			ei[i] = 1.0

			for k in range(self.n-1,-1,-1):

				vk = V[k:self.m,k]
				ei[k:self.m] = ei[k:self.m]-2.0*np.dot(np.outer(vk,np.conjugate(vk)),ei[k:self.m])

			self.Q[:,i] = ei


	def check_Q_orthogonal(self, tolerance):

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>10.8f}{:>+10.8f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else ('    {:>10.8f}    '.format(x.real) if x.imag == 0.0 else'     {:>10.8f}i   '.format(x.imag)))})

		QTQ = np.dot(np.transpose(self.Q),self.Q)
		dim = np.size(QTQ,0)
		print("\nQ^T Q = ",QTQ[0,:])
		for i in range(1,dim):
			print(" "*8,QTQ[i,:])
		print("")

		if np.allclose(QTQ,np.eye(dim),rtol=0.0,atol=tolerance):
			print("Q is orthogonal (tolerance = ", tolerance, ")")
		else:
			print("Q is not orthogonal (tolerance = ", tolerance, ")")




	# solve Ax=b for x, assuming A=QR has already been calculated
	def solve(self, b):

		# compute y=Q*b
		y = np.dot(np.transpose(np.conjugate(self.Q)),b)

		# solve Rx=y via back substitution
		x = np.zeros(self.n,dtype=complex)
		for i in range(self.n-1,-1,-1):
			x[i] = y[i]
			for j in range(i+1,self.n):
				x[i] += -self.R[i,j]*x[j]
			if(abs(self.R[i,i])>self.tolerance):
				x[i] = x[i]/self.R[i,i]

		print("\n","x = ", x[0])
		for i in range(1,np.size(x)):
			print(" "*5,x[i])

		print("\n","||Ax-b|| = ",np.linalg.norm(np.dot(self.A,x)-b))

		return x,np.linalg.norm(np.dot(self.A,x)-b)

		'''
		# display result
		names = ["b","x","Ax-b"]
		vectors = [b,x,np.dot(self.A,x)-b]
		dims = [self.m,self.n,self.m]
		widths = [6,6,9]

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>10.8f}{:>+10.8f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else ('    {:>10.8f}    '.format(x.real) if x.imag == 0.0 else'     {:>10.8f}i   '.format(x.imag)))})

		for n in range(len(names)):
			print("\n", names[n]," = ",vectors[n][0])
			for i in range(1,dims[n]):
				print(" "*widths[n],vectors[n][i])
		'''



	def compare(self, b):

		# over-determined systems
		if self.m >= self.n:

			print("\nClassical Gram-Schmidt Method:")
			self.classical_GramSchmidt()
			self.display_result()
			self.check_Q_orthogonal(1E-10)
			self.solve(b)

			print("\nModified Gram-Schmidt Method:")
			self.modified_GramSchmidt()
			self.display_result()
			self.check_Q_orthogonal(1E-10)
			self.solve(b)

			print("\nHouseholder Transform Based Method:")
			self.Householder()
			self.display_result()
			self.check_Q_orthogonal(1E-10)
			self.solve(b)

		# under-determined systems
		else:

			# QR factorize A* instead of A
			self.A = np.transpose(np.conjugate(self.A))
			self.m, self.n = self.A.shape

			# full QR factorization
			print("\nHouseholder Transform Based Method:")
			self.Householder()

			# extract R*
			R_star = np.transpose(np.conjugate(self.R[:self.n,:self.n]))
			
			# forward substitution to solve R*y_R = b
			y = np.zeros(self.m,dtype=complex)
			for i in range(self.n):
				y[i] = b[i]
				for j in range(i):
					y[i] += -R_star[i,j]*y[j]
				if (abs(R_star[i,i])>self.tolerance):
					y[i] = y[i]/R_star[i,i]

			# solve for x
			x = np.dot(self.Q,y)
			
			# display
			self.display_result()
			self.check_Q_orthogonal(1E-10)

			np.set_printoptions(formatter={'complexfloat':lambda x:('{:>10.8f}{:>+10.8f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else ('    {:>10.8f}    '.format(x.real) if x.imag == 0.0 else'     {:>10.8f}i   '.format(x.imag)))})

			print("\n","x = ", x[0])
			for i in range(1,np.size(x)):
				print(" "*5,x[i])

			print("\n","||Ax-b|| = ",np.linalg.norm(np.dot(np.transpose(np.conjugate(self.A)),x)-b))



	def display_result(self):

		names = ["A","Q","R","A-QR"]
		matrices = [self.A,self.Q,self.R,self.A-np.dot(self.Q,self.R)]
		dims = [np.size(matrices[n],0) for n in range(4)]
		widths = [6,6,6,9]

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>10.8f}{:>+10.8f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else ('    {:>10.8f}    '.format(x.real) if x.imag == 0.0 else'     {:>10.8f}i   '.format(x.imag)))})

		for n in range(len(names)):
			print("\n", names[n]," = ",matrices[n][0,:])
			for i in range(1,dims[n]):
				print(" "*widths[n],matrices[n][i,:])






				