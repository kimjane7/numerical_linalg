import os
import sys
import numpy as np
from numpy import linalg
from scipy.linalg import expm, eigvals, qr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.family'] = "serif"


class SL_Solver:

	def __init__(self, N, p, q):

		self.p = p
		self.q = q
		self.N = N
		self.h = np.pi/N
		self.tol = 1E-15

		self.x = np.zeros(N+1)                     # grid points
		self.u_exact = np.zeros((N+1,N))           # exact eigenvectors
		self.lambda_exact = np.zeros(N)            # exact eigenvalues
		self.lambdah_hat = np.zeros(N)             # A scheme eigenvalues
		self.lambdah = np.zeros(N)                 # B scheme eigenvalues

		for i in range(N+1):

			self.x[i] = i*self.h

			for j in range(N):

				kh = 2.0*(1.0-np.cos((j+0.5)*self.h))/(self.h**2)
				mh = (2.0+np.cos((j+0.5)*self.h))/3.0

				self.u_exact[i,j] = np.sqrt(2.0/np.pi)*np.sin((j+0.5)*self.x[i])
				self.lambda_exact[j] = self.p*(j+0.5)**2+self.q
				self.lambdah[j] = (self.p*kh+self.q*mh)/mh
				self.lambdah_hat[j] = self.p*kh+q


	def inversepower(self, A):

		# store eigenvectors and eigenvalues for each iteration
		kmax = 1000
		Lambda = np.zeros(kmax)
		V = np.zeros((self.N,kmax))

		# initial guess
		Lambda[0] = 0.0
		V0 = np.ones(self.N)
		V[:,0] = V0/np.linalg.norm(V0)
		Lambda_diff = 10.0
		k = 1

		while (Lambda_diff > self.tol) and (k < kmax):
			W = np.linalg.solve(A,V[:,k-1])
			V[:,k] = W/np.linalg.norm(W)
			Lambda[k] = np.dot(np.dot(np.transpose(V[:,k]),A),V[:,k])
			Lambda_diff = abs(Lambda[k]-Lambda[k-1])
			k += 1

		print(k)

		return Lambda[k-1]


	def shiftedpower(self, A):

		# store eigenvectors and eigenvalues for each iteration
		kmax = 500000
		Lambda = np.zeros(kmax)
		V = np.zeros((self.N,kmax))
		I = np.eye(self.N)

		'''
		# use power method to find largest eigenvalue
		Lambda[0] = 0.0
		V0 = np.ones(self.N)
		V[:,0] = V0/np.linalg.norm(V0)
		Lambda_diff = 10.0
		k = 1

		while (Lambda_diff > self.tol) and (k < kmax):
			W = np.dot(A,V[:,k-1])
			V[:,k] = W/np.linalg.norm(W)
			Lambda[k] = np.dot(np.dot(np.transpose(V[:,k]),A),V[:,k])
			Lambda_diff = abs(Lambda[k]-Lambda[k-1])
			k += 1

		print("k = ", k)
		print("largest eigval = ", self.p*Lambda[k-1]/(self.h**2)+self.q)

		'''

		# shift by largest eigenvalue
		#mu = Lambda[k-1]
		mu = self.lambda_exact[-1]
		V0 = np.ones(self.N)
		V[:,0] = V0/np.linalg.norm(V0)
		Lambda_diff = 10.0
		k = 1
		while (Lambda_diff > self.tol) and (k < kmax):
			W = np.dot(A-mu*I,V[:,k-1])
			V[:,k] = W/np.linalg.norm(W)
			Lambda[k] = np.dot(np.dot(np.transpose(V[:,k]),A),V[:,k])
			Lambda_diff = abs(Lambda[k]-Lambda[k-1])
			k += 1

		print("smallest eigval = ", self.p*Lambda[k-1]/(self.h**2)+self.q)
		print("k = ", k)

		return Lambda[k-1]


	def schemeA(self, method):


		Lambda = 0.0

		# form scheme A matrix
		A = np.zeros((self.N,self.N))

		A[0,1] = -1.0
		A[self.N-1,self.N-2] = -2.0
		for i in range(self.N):
			A[i,i] = 2.0
			if (0 < i < self.N-1):
				A[i,i-1] = -1.0
				A[i,i+1] = -1.0



		# find the smallest eigenvalue of A
		if method == 'inverse power':
			Lambda = self.inversepower(A)

		# find the smallest eigenvalue of A
		if method == 'shifted power':
			Lambda = self.shiftedpower(A)




		# find all eigenvalues of A using QR iteration with deflation and shifts
		if method == 'QR':
			while (abs(Lambda[k]-Lambda[k-1]) > tolerance) and (k < kmax):
				print(k)
				print(A)
				# shifted QR
				Q,R = np.linalg.qr(A-mu*I)
				A = np.dot(R,Q)+mu*I

				# zero out small lower diagonal elements
				for i in range(1,self.N):
					if abs(A[i,i-1]) < tolerance:
						A[i,i-1] = 0.0
						print(i)

				# pick new shift
				Lambda[k] = A[self.N-1,self.N-1]
				mu = Lambda[k]
				print("mu = ",mu)
				k += 1

	
		print(self.N)
		self.schemeA_eigval = self.p*Lambda/(self.h**2)+self.q
		print(self.schemeA_eigval)


	def plot_eigvec(self, V):

		# enforce boundary condition at 0
		if (V.size == self.N):
			V = np.concatenate([[0.0],V], axis=None)

		plt.figure(figsize=(8,6))
		plt.plot(self.x, 6*V,color='blue')
		plt.plot(self.x, self.u_exact[:,0],color='red')

		figname = 'plot.png'
		plt.savefig(figname, format='png')
		os.system('okular '+figname)
		plt.clf()







