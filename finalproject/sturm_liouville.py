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

		return Lambda_new, count


	def shiftedpower(self, A):

		# shift
		mu = A[0,0]+1E-3
		I = np.eye(self.N)
		B = A-mu*I

		# initial guess
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


		return Lambda_new, count


	def qr_deflation(self, A):

		all_eigvals = np.zeros(self.N)
		B = A.copy()
		I = np.eye(self.N)

		count = 0
		for n in range(self.N-1,0,-1):

			while abs(B[n,n-1]) > self.tol:

				Q,R = self.qr(B)
				B = np.dot(R,Q) 
				count += 1

			all_eigvals[n] = B[n,n]
			B = B[:n,:n]
			I = np.eye(n)

		all_eigvals[0] = B[0,0]
		all_eigvals = np.sort(all_eigvals)


		return all_eigvals[0], count


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
			Lambda, count = self.inversepower(A)

		if method == 'shifted power':
			Lambda, count = self.shiftedpower(A)

		if method == 'QR':
			Lambda, count = self.qr_deflation(A)

		self.schemeA_eigval = self.p*Lambda/(self.h**2)+self.q

		print("{:^10d}{:^20d}{:^20f}\n".format(self.N, count, self.schemeA_eigval))




	def schemeB(self, method):

		Lambda = 0.0

		# form scheme B matrices
		A = np.zeros((self.N,self.N))
		A[0,1] = -1.0
		A[self.N-1,self.N-2] = -2.0
		for i in range(self.N):
			A[i,i] = 2.0
			if (0 < i < self.N-1):
				A[i,i-1] = -1.0
				A[i,i+1] = -1.0

		B = np.zeros((self.N,self.N))
		B[0,1] = 1.0
		B[self.N-1,self.N-2] = 2.0
		for i in range(self.N):
			B[i,i] = 4.0
			if (0 < i < self.N-1):
				B[i,i-1] = 1.0
				B[i,i+1] = 1.0

		# get QR decomposition of B
		Q,R = self.qr(B)
		QT = np.transpose(Q)
		Rinv = np.linalg.inv(R)
		QAR = np.dot(QT,np.dot(A,Rinv))

		# choose method
		if method == 'inverse power':
			Lambda, count = self.inversepower(QAR)

		if method == 'shifted power':
			Lambda, count = self.shiftedpower(QAR)

		if method == 'QR':
			Lambda, count = self.qr_deflation(QAR)

		self.schemeB_eigval = 6.0*self.p*Lambda/(self.h**2)+self.q

		print("{:^10d}{:^20d}{:^20f}\n".format(self.N, count, self.schemeB_eigval))


	def qr(self, A):

		m = A.shape[0]
		Q = np.zeros_like(A)
		R = np.zeros_like(A)
		V = A.copy()

		for i in range(m):

			R[i,i] = np.linalg.norm(V[:,i])

			if R[i,i] > self.tol:
				Q[:,i] = V[:,i]/R[i,i]

			for j in range(i+1,m):
				R[i,j] = np.vdot(Q[:,i],V[:,j])
				V[:,j] -= R[i,j]*Q[:,i]

		return Q, R
