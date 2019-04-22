import os
import sys
import numpy as np
from numpy import linalg
from scipy.linalg import expm, eigvals
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.rcParams['font.family'] = "serif"


class SL_Solver:

	def __init__(self, N, jmax, p, q):

		self.p = p
		self.q = q
		self.N = N
		self.h = np.pi/N

		self.x = np.zeros(N+1)                        # grid points
		self.u_exact = np.zeros((N+1,jmax))            # exact eigenvectors
		self.lambda_exact = np.zeros(jmax)            # exact eigenvalues
		self.lambdah_hat = np.zeros(jmax)             # A scheme eigenvalues
		self.lambdah = np.zeros(jmax)                 # B scheme eigenvalues

		for i in range(N+1):

			self.x[i] = i*self.h

			for j in range(jmax):

				kh = 2.0*(1.0-np.cos((j+0.5)*self.h))/(self.h**2)
				mh = (2.0+np.cos((j+0.5)*self.h))/3.0

				self.u_exact[i,j] = np.sqrt(2.0/np.pi)*np.sin((j+0.5)*self.x[i])
				self.lambda_exact[j] = self.p*(j+0.5)**2+self.q
				self.lambdah[j] = (self.p*kh+self.q*mh)/mh
				self.lambdah_hat[j] = self.p*kh+q


	def schemeA(self, V0, method, kmax, mu = 0.0):

		tolerance = 1E-12

		# identity
		I = np.eye(self.N)

		# form scheme A matrix
		A = np.zeros((self.N,self.N))

		A[0,1] = -1.0
		A[self.N-1,self.N-2] = -2.0
		for i in range(self.N):
			A[i,i] = 2.0
			if (0 < i < self.N-1):
				A[i,i-1] = -1.0
				A[i,i+1] = -1.0

		# store eigenvectors and eigenvalues for each iteration
		Lambda = np.zeros(kmax)
		V = np.zeros((self.N,kmax))

		# normalize initial guess of eigenvector
		V[:,0] = V0/np.linalg.norm(V0)

		# initial estimate for associated eigenvalue
		Lambda[0] = 0.0
		Lambda[-1] = 100.0
		k = 0

		# find the eigenvalue of A that is closest to mu
		if method == 'inverse power':
			while (abs(Lambda[k]-Lambda[k-1]) > tolerance) and (k < kmax):
				W = np.dot(np.linalg.inv(A-mu*I),V[:,k-1])
				V[:,k] = W/np.linalg.norm(W)
				Lambda[k] = np.dot(np.dot(np.transpose(V[:,k]),A),V[:,k])
				k += 1

		# find the eigenvalue of A that is furthest from mu
		if method == 'shifted power':
			while (abs(Lambda[k]-Lambda[k-1]) > tolerance) and (k < kmax):
				W = np.dot(A-mu*I,V[:,k-1])
				V[:,k] = W/np.linalg.norm(W)
				Lambda[k] = np.dot(np.dot(np.transpose(V[:,k]),A),V[:,k])
				k += 1

		# find all eigenvalues of A using QR iteration with deflation and Wilkinson shift
		if method == 'QR':
			while (abs(Lambda[k]-Lambda[k-1]) > tolerance) and (k < kmax):
				mu = wilkinson

		print(k)


		self.schemeA_eigvals = self.p*Lambda[k]/(self.h**2)+self.q
		#self.plot_eigvec(V[:,kmax-1])


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







