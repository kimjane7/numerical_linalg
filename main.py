from QR_decomp import QR_Decomposition
import numpy as np 

def main():

	Z = np.array([[1,2,3],[4,5,6],[7,8,7],[4,2,3],[4,2,2]],dtype=complex)
	QR_Z = QR_Decomposition(Z)
	QR_Z.compare("case1")

	A = np.array([[0.70000,0.70711],[0.70001,0.70711]],dtype=complex)
	QR_A = QR_Decomposition(A)
	QR_A.compare("case2")

main()