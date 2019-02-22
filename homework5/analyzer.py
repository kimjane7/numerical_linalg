import numpy as np 
from numpy import array, dot, diag
from randmat import RandomMatrix

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import numpy as np
import scipy.stats as stats
import os
import sys
from pylab import *
matplotlib.rcParams['font.family'] = "serif"
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter

class Analyzer:

	def __init__(self, m_list, sigma_min_bounds):

		self.m_list = m_list
		self.bounds = sigma_min_bounds

	def plot_eigenvalues(self, N):

		for m in self.m_list:

			# construct lists of N RandomMatrix objects
			full_randmats = [RandomMatrix(m) for i in range(N)]
			tri_randmats = [RandomMatrix(m,True) for i in range(N)]

			# superimpose 

			

