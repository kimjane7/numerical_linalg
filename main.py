from QR_decomp import QR_Decomposition
import numpy as np 
from scipy.linalg import hilbert

# Tikhonov regularization
def Hilbert_test(order_list, alpha_list):

	print("{:<6s}{:^10s}{:^18s}{:^18s}{:^18s}{:^18s}{:^18s}{:^18s}\n".format("n","alpha","||Ax-b|| (cGS)","||x|| (cGS)","||Ax-b|| (mGS)","||x|| (mGS)","||Ax-b|| (H)","||x|| (H)"))

	for n in order_list:

		b_reg = np.zeros(2*n,dtype=complex)
		for i in range(n):
			b_reg[i] = 1.0; 

		H = hilbert(n)
		H_reg = np.zeros((2*n,n),dtype=complex)
		H_reg[:n,:n] = H

		for alpha in alpha_list:

			H_reg[n:,:n] = alpha*np.eye(n,dtype=complex)

			QR_H_reg = QR_Decomposition(H_reg)

			QR_H_reg.classical_GramSchmidt()
			x1, norm1 = QR_H_reg.solve(b_reg)

			QR_H_reg.modified_GramSchmidt()
			x2, norm2 = QR_H_reg.solve(b_reg)

			QR_H_reg.Householder()
			x3, norm3 = QR_H_reg.solve(b_reg)

			print("{:<6d}{:^10.1e}{:^18.8f}{:^18.8f}{:^18.8f}{:^18.8f}{:^18.8f}{:^18.8f}".format(n,alpha,norm1,np.linalg.norm(x1),norm2,np.linalg.norm(x2),norm3,np.linalg.norm(x3)))

		print(" ")
				

def main():

	#Hilbert_test([10,20,30],[1.0E-8,1.0E-5,1.0E-3,0.1])

	
	# 2 (a)
	A = np.array([[1,2,3],[4,5,6],[7,8,7],[4,2,3],[4,2,2]],dtype=complex)
	QR_A = QR_Decomposition(A)
	QR_A.compare(np.array([1,1,1,1,1],dtype=complex))
	
	# 2 (b)
	A = np.array([[0.70000,0.70711],[0.70001,0.70711]],dtype=complex)
	QR_A = QR_Decomposition(A)
	QR_A.compare(np.array([1,1],dtype=complex))

	# 2 (c)
	A = np.array([[1,2,3],[4,2,9]],dtype=complex)
	QR_A = QR_Decomposition(A)
	QR_A.compare(np.array([6,15],dtype=complex))
	



main()