#################################################
############### solves Ax=b for x ###############
#################################################

import numpy as np 
from numpy import array, dot, outer, diag, reshape, transpose, linalg

class Solver:

	def __init__(self, A, b, A_name):

		self.m = A.shape[0]
		self.A = A.astype(complex)
		self.b = b.astype(complex)
		self.A_name = A_name

		self.calc_kappa()
		print("kappa = ", self.kappa)


	# condition number of A
	def calc_kappa(self):

		self.kappa = np.linalg.norm(self.A)*np.linalg.norm(np.linalg.inv(self.A))


	# use LU decomposition (A=LU) without pivoting to solve linear system
	def LU(self):

		L = np.eye(self.m, dtype=complex)
		U = self.A.copy()

		for k in range(self.m-1):

			L[k+1:self.m,k] = U[k+1:self.m,k]/U[k,k]
			U[k+1:self.m,k:self.m] -= np.outer(L[k+1:self.m,k],U[k,k:self.m])

		self.forward_sub(L)
		self.backward_sub(U)
		self.display_result("L","U",L,U)


	# use Cholesky factorization (A=R*R) to solve linear system 
	def cholesky(self):

		R = self.A.copy()

		for k in range(self.m):
			for j in range(k+1, self.m):
				R[j,j:self.m] -= R[k,j:self.m]*np.conj(R[k,j])/R[k,k]
			R[k,k:self.m] /= np.sqrt(R[k,k])

		self.forward_sub(np.conjugate(np.transpose(R)))
		self.backward_sub(R)
		self.display_result("R*","R",np.conjugate(np.transpose(R)),R)


	# use modified Gram-Schmidt QR method (A=QR) to solve linear system
	def QR(self):

		Q = np.zeros((self.m,self.m),dtype=complex)
		R = np.zeros((self.m,self.m),dtype=complex)
		V = self.A.copy()
		tol = 1E-30

		for i in range(self.m):

			R[i,i] = np.linalg.norm(V[:,i])

			if np.abs(R[i,i]) > tol:
				Q[:,i] = V[:,i]/R[i,i]

			for j in range(i+1,self.m):
				R[i,j] = np.vdot(Q[:,i],V[:,j])
				V[:,j] -= dot(R[i,j],Q[:,i])

		self.y = np.dot(np.transpose(np.conjugate(Q)),self.b)
		self.backward_sub(R)
		self.display_result("Q","R",Q,R)


	# solve Ly=b for y
	def forward_sub(self, L):

		self.y = np.zeros(self.m, dtype=complex)

		self.y = self.b.copy()
		for i in range(self.m):
			self.y[i] -= np.dot(L[i,:i],self.y[:i])
			self.y[i] /= L[i,i]		


	# solve Ux=y for x
	def backward_sub(self, U):

		self.x = np.zeros(self.m, dtype=complex)

		self.x = self.y.copy()
		for i in range(self.m-1,-1,-1):
			self.x[i] -= np.dot(U[i,i+1:],self.x[i+1:])
			self.x[i] /= U[i,i]


	def display_result(self, L_name, U_name, L, U):

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>4.3f}{:>+4.3f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else (' {:>4.3f} '.format(x.real) if x.imag == 0.0 else' {:>4.3f}i'.format(x.imag)))})

		matrix_names = [self.A_name, L_name, U_name, self.A_name+"-"+L_name+U_name]
		matrices = [self.A, L, U, self.A-np.dot(L,U)]
		widths = [9,6,6,12]

		for n in range(len(matrices)):
			print("\n", matrix_names[n], " = ", matrices[n][0,:])
			for i in range(1,self.m):
				print(" "*widths[n],matrices[n][i,:])

		vector_names = ["x", "b", self.A_name+"x-b"]
		vectors = [self.x, self.b, np.dot(self.A,self.x)-self.b]
		widths = [6,6,12]

		for n in range(len(vectors)):
			print("\n", vector_names[n], " = ", vectors[n][0])
			for i in range(1,self.m):
				print(" "*widths[n],vectors[n][i])

		print("\n||"+self.A_name+"x-b|| = ",np.linalg.norm(vectors[-1]))



				