import numpy as np 



# 3

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = "serif"

plt.figure(figsize=(8,6))
x = np.arange(1.920,2.081,0.001)
p = x**9-18*x**8+144*x**7-672*x**6+2016*x**5-4032*x**4+5376*x**3-4608*x**2+2304*x-512
plt.plot(x,p,c='r')

p = (x-2)**9
plt.plot(x,p,c='r')

plt.show()


# 4

b = np.float32(1.0)
c = np.float32(0.004004)

a = 1000*(c/(np.sqrt(b**2+c)-b)-2*b)
print(a)

a = 1000*c/(np.sqrt(b**2+c)+b)
print(a)