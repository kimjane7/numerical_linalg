import numpy as np
from numpy import linalg
from scipy.linalg import qr


# reduces real symmetric matrix A into tridiagonal form (Algo. 26.1)
def tridiag(A):

	tolerance = 1E-12
	m = A.shape[0]
	T = A.copy()


	for k in range(m-2):

		x = T[k+1:m,k]

		e1 = np.zeros(m-k-1)
		e1[0] = 1.0

		v = np.sign(x[0])*np.linalg.norm(x)*e1+x
		v = v/np.linalg.norm(v)

		T[k+1:m,k:m] -= 2.0*np.outer(v,np.dot(np.conjugate(v),T[k+1:m,k:m]))
		T[0:m,k+1:m] -= 2.0*np.outer(np.dot(T[0:m,k+1:m],v),np.conjugate(v))

	for i in range(m):
		for j in range(m):
			if abs(T[i,j]) < tolerance:
				T[i,j] = 0.0

	return T


# runs unshifted QR algorithm on a real tridiagonal matrix T (Algo. 28.1 & 28.2)
def qralg(T, Wilkinson = False):

	tolerance = 1E-12
	m = T.shape[0]
	Tnew = T.copy()
	t = np.array([])


	if Wilkinson:

		while abs(Tnew[m-2,m-1]) > tolerance:

			t = np.append(t,abs(Tnew[m-2,m-1]))

			# Wilkinson shift
			delta = 0.5*(Tnew[m-2,m-2]-Tnew[m-1,m-1])
			sign = np.sign(delta)
			if delta == 0.0:
				sign = 1.0
			mu = Tnew[m-1,m-1]-sign*Tnew[m-2,m-1]**2/(abs(delta)+np.sqrt(delta**2+Tnew[m-2,m-1]**2))
			I = np.eye(Tnew.shape[0])

			# shifted QR
			Q,R = np.linalg.qr(Tnew-mu*I)
			Tnew = np.dot(R,Q)+mu*I

	else:

		while abs(Tnew[m-2,m-1]) > tolerance:

			t = np.append(t,abs(Tnew[m-2,m-1]))

			# pure QR
			Q,R = np.linalg.qr(Tnew)
			Tnew = np.dot(R,Q)

	t = np.append(t,abs(Tnew[m-2,m-1]))
	for i in range(m):
		for j in range(m):
			if abs(Tnew[i,j]) < tolerance:
				Tnew[i,j] = 0.0

	return Tnew, t