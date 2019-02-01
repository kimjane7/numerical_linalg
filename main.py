from QR_decomp import QR_Decomposition
import numpy as np 
from numpy import array

def main():

	#Z = np.array([[1,2,3],[4,5,6],[7,8,7],[4,2,3],[4,2,2]])
	#Z = np.array([[0+1j*1, 2-1j*3],[-2+1j*1,-4-1j*8]])
	Z = np.array([[1,1,0],[1,0,1],[0,1,1]])

	solver = QR_Decomposition(Z)

main()