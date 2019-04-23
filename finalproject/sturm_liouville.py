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

	def __init__(self, N, p, q, tolerance):

		self.p = p
		self.q = q
		self.N = N
		self.h = np.pi/N
		self.tol = tolerance

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

		# initial guess
		V_old = np.ones(self.N)
		V_old = V_old/np.linalg.norm(V_old)
		Lambda_old = 0.0
		Lambda_diff = 1.0

		# iteration
		count = 0
		while Lambda_diff > self.tol:

			V_new = np.linalg.solve(A,V_old)
			V_new = V_new/np.linalg.norm(V_new)
			Lambda_new = np.dot(np.dot(np.transpose(V_new),A),V_new)
			Lambda_diff = abs(Lambda_new-Lambda_old)

			V_old = V_new
			Lambda_old = Lambda_new
			count += 1

		print("{:<d} {:<d} {:<f}\n".format(self.N, count, self.p*Lambda_new/(self.h**2)+self.q))

		return Lambda_new


	def shiftedpower(self, A):

		mu = A[0,0]+1E-3

		'''
		n = np.log2(self.N)
		mu = A[0,0]+5**(-n)
		'''

		I = np.eye(self.N)
		B = A-mu*I

		V_old = np.ones(self.N)
		V_old = V_old/np.linalg.norm(V_old)
		Lambda_old = 0.0
		Lambda_diff = 1.0
		
		# iteration
		count = 0
		while Lambda_diff > self.tol:

			V_new = np.dot(B,V_old)
			V_new = V_new/np.linalg.norm(V_new,2)
			Lambda_new = np.dot(np.dot(np.transpose(V_new),A),V_new)
			Lambda_diff = abs(Lambda_new-Lambda_old)
			
			V_old = V_new
			Lambda_old = Lambda_new
			count += 1

		print("{:<d} {:<d} {:<f} {:<f}\n".format(self.N, count, self.p*Lambda_new/(self.h**2)+self.q, mu))

		return Lambda_new


	def qr_deflation(self, A):

		count = 0

		while abs(A[1,0]) > self.tol:

			Q,R = np.linalg.qr(A)
			A = np.dot(R,Q)
			count += 1

		for i in range(1,self.N):
			if abs(A[i,i-1]) < self.tol:
				A[i,i-1] = 0.0


		print(count)
		print(A)

		return 0




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

		# choose method
		if method == 'inverse power':
			Lambda = self.inversepower(A)

		if method == 'shifted power':
			Lambda = self.shiftedpower(A)

		if method == 'QR':
			Lambda = self.qr_deflation(A)

		self.schemeA_eigval = self.p*Lambda/(self.h**2)+self.q


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







