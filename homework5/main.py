import numpy as np
from analyzer import Analyzer 

def main():

	m_list = [2**n for n in range(3,6)]
	sigma_min_bounds = [0.5**n for n in range(1,7)]

	A = Analyzer(m_list,sigma_min_bounds)








main()