import numpy as np 
from numpy import array, dot, diag, reshape, transpose, linalg
from scipy.linalg import eigvalsh, expm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

class QR_Decomposition:


	def __init__(self, A):

		self.m, self.n = A.shape
		self.A = A
		self.tolerance = 1E-30

	def classical_GramSchmidt(self):

		self.Q = np.zeros((self.n,self.n),dtype=complex)
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

		self.Q = np.zeros((self.m,self.n),dtype=complex)
		self.R = self.A.copy()
		V = np.zeros((self.m,self.n),dtype=complex)

		for k in range(self.n):

			# householder transformation
			x = self.R[k:self.m,k]
			e1 = np.zeros(self.m-k)
			e1[0] = 1.0
			vk = np.sign(x[0])*np.linalg.norm(x)*e1+x
			if np.linalg.norm(vk) > self.tolerance:
				vk = vk/np.linalg.norm(vk)

			# store to calculate Q
			V[k:self.m,k] = vk

			# calculate R		
			self.R[k:self.m,k:self.n] = np.dot(np.eye(self.m-k)-2.0*np.outer(vk,np.conjugate(vk)),self.R[k:self.m,k:self.n])


		# calculate Q*I
		for i in range(self.n):

			ei = np.zeros(self.m,dtype=complex)
			ei[i] = 1.0

			for k in range(self.n-1,-1,-1):

				vk = V[k:self.m,k]
				ei[k:self.m] = ei[k:self.m]-2.0*np.dot(np.outer(vk,np.conjugate(vk)),ei[k:self.m])

			self.Q[:,i] = ei



	def check_Q_orthogonal(self,tolerance):

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>7.5f}{:>+7.5f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else ('    {:>7.5f}    '.format(x.real) if x.imag == 0.0 else'     {:>7.5f}i   '.format(x.imag)))})

		QTQ = np.dot(np.transpose(self.Q),self.Q)
		print("\nQ^T Q = ",QTQ[0,:])
		for i in range(1,self.n):
			print(" "*8,QTQ[i,:])
		print("")

		if np.allclose(QTQ,np.eye(self.n),rtol=0.0,atol=tolerance):
			print("Q is orthogonal (tolerance = ", tolerance, ")")
		else:
			print("Q is not orthogonal (tolerance = ", tolerance, ")")


	def compare(self,filename):

		self.classical_GramSchmidt()
		self.display_result()
		self.check_Q_orthogonal(1E-10)

		self.modified_GramSchmidt()
		self.display_result()
		self.check_Q_orthogonal(1E-10)

		self.Householder()
		self.display_result()
		self.check_Q_orthogonal(1E-10)



	def display_result(self):

		names = ["A","Q","R","A-QR"]
		matrices = [self.A,self.Q,self.R,self.A-np.dot(self.Q,self.R[:self.n,:self.n])]
		dims = [self.m,self.m,self.n,self.m]
		widths = [6,6,6,9]

		np.set_printoptions(formatter={'complexfloat':lambda x:('{:>7.5f}{:>+7.5f}i'.format(x.real,x.imag) if (x.real != 0.0 and x.imag != 0.0) else ('    {:>7.5f}    '.format(x.real) if x.imag == 0.0 else'     {:>7.5f}i   '.format(x.imag)))})

		for n in range(len(names)):
			print("\n", names[n]," = ",matrices[n][0,:])
			for i in range(1,dims[n]):
				print(" "*widths[n],matrices[n][i,:])

