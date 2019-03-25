import os
import sys
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = "serif"


# 3

plt.figure(figsize=(8,6))
x = np.arange(1.920,2.081,0.001)

p = (x-2)**9
plt.plot(x,p,c='b',label=r'factorized $p(x)=(x-2)^9$')


p = x**9-18*x**8+144*x**7-672*x**6+2016*x**5-4032*x**4+5376*x**3-4608*x**2+2304*x-512
plt.plot(x,p,c='r',label=r'expanded $p(x)$')



plt.xlabel(r'$x$',fontsize=12)
plt.legend(loc='upper left', shadow=True, fontsize=12)
plt.title(r'Stability of a polynomial')

# save and open
figname = 'polynomial.png'
plt.savefig(figname, format='png')
os.system('okular '+figname)
plt.clf()


# 4

b = np.float32(1.0)
c = np.float32(0.004004)

a = 1000*(c/(np.sqrt(b**2+c)-b)-2*b)
print(a)

a = 1000*c/(np.sqrt(b**2+c)+b)
print(a)