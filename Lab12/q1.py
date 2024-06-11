import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
w = pd.read_csv('wiene.csv', sep=',', header=None)
p = pd.read_csv('price.csv', sep=',', header=None)
g = pd.read_csv('gauss.csv', sep=',', header=None)
f = pd.read_csv('fluct.csv', sep=',', header=None)
t = pd.read_csv('trade.csv', sep=',', header=None)

# Price data
pc3 = p.iloc[:, 2].values
pc2 = p.iloc[:, 1].values

a = 0.0005
ti = np.arange(1, 5562)
s = lambda t: 800 * np.exp(a * t)
plt.figure(figsize=(10, 6))
plt.semilogy(pc2, pc3, ti, s(ti))
plt.title('Price Data')
plt.xlim([1, 5561])
plt.ylabel('S')
plt.xlabel('t')
plt.legend(["Actual Data", "Theoretical"], loc="upper right")
plt.savefig('price.png')
plt.show()

# Fluctuation data
plt.figure(figsize=(10, 6))
fl3 = f.iloc[:, 2].values
fl2 = f.iloc[:, 1].values
plt.plot(fl2, fl3)
plt.ylabel('\u03B4')
plt.title('Fluctuation Data')
plt.xlabel('t')
plt.xlim([1, 5561])
plt.legend([''])
plt.savefig('fluctuation.png')
plt.show()

# Gaussian data
u = 0.057
yp = 1.495
fo = 18
plt.figure(figsize=(10, 6))
gaussian_func = lambda dt: 1 + fo * (np.exp(-1 * ((dt - u) ** 2) / (2 * yp * yp)))
gs3 = g.iloc[:, 2].values
gs2 = g.iloc[:, 1].values
ti = np.arange(-13, 18, 0.001)
plt.title('Gaussian Data')
plt.plot(gs2, gs3, ti, gaussian_func(ti), '--')
plt.xlim([-10, 10])
plt.ylabel('f(\u03B4)')
plt.xlabel('\u03B4')
plt.legend(["Actual Data", "Theoretical"], loc="upper right")
plt.savefig('gaussian.png')
plt.show()

# Wiene data
wn1 = w.iloc[:, 0].values
wn3 = w.iloc[:, 2].values
plt.figure(figsize=(10, 6))
ti = np.arange(0, 250, 0.01)
linear_func = lambda x: 0.01 * x + 6.75
plt.plot(wn1, wn3, ti, linear_func(ti))
plt.ylabel('ln(S)')
plt.title('Wiene Data')
plt.xlabel('\u03C4')
plt.legend(["Actual Data", "Theoretical"], loc="upper right")
plt.savefig('wiene.png')
plt.show()

wn4 = w.iloc[:, 3].values
plt.figure(figsize=(10, 6))
ti = np.arange(0, 250, 0.01)
linear_func = lambda x: -3.41e-6 * x + 0.00113
plt.plot(wn1, wn4, ti, linear_func(ti))
plt.ylabel('\u03A3^2')
plt.xlabel('\u03C4')
plt.title('Wiener Variance Data')
plt.legend(["Actual Data", "Theoretical"], loc="upper right")
plt.savefig('wiene2.png')
plt.show()

# Trade data
tr2 = t.iloc[:, 1].values
tr3 = t.iloc[:, 2].values
a = 0.0004
ti = np.arange(1, 5562)
s = lambda t: 34 * np.exp(a * t)
plt.figure(figsize=(10, 6))
plt.semilogy(tr2, tr3, ti, s(ti))
plt.title('Trade Data')
plt.ylabel('N')
plt.xlabel('t')
plt.legend(["Actual Data", "Theoretical"], loc="upper right")
plt.savefig('trade.png')
plt.show()
