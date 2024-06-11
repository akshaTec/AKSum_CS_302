import numpy as np
import matplotlib.pyplot as plt

D = 1

def f(x, t):
    return (4 * np.pi * D * t) ** (-0.5) * (np.exp((-x ** 2) / (4 * D * t)))

def y(x, t):
    a = 1
    v = 1
    yo = 1
    return yo * (np.exp(-a * ((x - v * t) ** 2)))

x = np.arange(-4, 4, 0.01)
t1, t2, t3 = 0.01, 0.1, 1

plt.figure(figsize=(10, 6))

# plt.subplot(2, 2, 1)
plt.plot(x, f(x, t1), label='t=0.01')
plt.plot(x, f(x, t2), label='t=0.1')
plt.plot(x, f(x, t3), label='t=1')

plt.xlim([-4, 4])
plt.legend()
plt.xlabel('x')
plt.ylabel('Ψ(x,t)')
plt.grid(True)
plt.show()

t = np.arange(0.01, 1, 0.001)
# plt.subplot(2, 2, 2)
plt.plot(t, f(0, t))
plt.xlim([0, 1])
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Ψ(0,t)')
plt.show()

# plt.subplot(2, 2, 3)
plt.plot(t, f(0, t))
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('Ψ(0,t)')
plt.show()

# plt.subplot(2, 2, 4)
plt.plot(x, y(x, t1), label='t=0.01')
plt.plot(x, y(x, t2), label='t=0.1')
plt.plot(x, y(x, t3), label='t=1')
plt.xlim([-4, 4])
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y(x,t)')
plt.legend()
plt.show()




