import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

t = 1000

def sin_accretion_rate(ts, M0, n):
    f = lambda ts: M0 + np.sin((ts/t)*4*np.pi/n)*n
    ys = f(ts)
    n = n/10
    return ys, n

n = 1
ts = np.arange(0, t)
M0 = np.zeros(t)
Ar, n = sin_accretion_rate(ts, M0, n)
Ar, n = sin_accretion_rate(ts, Ar, n)

plt.plot(ts, Ar)
plt.show()
freq, psd = signal.welch(Ar)
plt.plot(freq, psd)
print(psd)