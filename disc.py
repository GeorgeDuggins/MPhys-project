import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

R = 10000 # radius of disk in units of Gravitational radii Rg = GM/c^2
a = 10 # number of annuli
R0 = 1 # inner edge of disc
alpha = 1 # viscocity perameter
H = 1 #disc height
t = 1000000 #number of time values at which we measure Mdot

def annuli_displacements(Radius, MinRad, Annuli):
    displacements = np.logspace(np.log10(Radius), np.log10(MinRad), Annuli)
    return displacements

def viscous_frequency(displacements, Radius, Height, alpha):
    f = lambda displacements: displacements**(-3/2)*((Height/Radius)**2)*(alpha/(2*np.pi))
    frequencies = f(displacements)
    return frequencies
    
def toy_accretion_rate(M0, r, ts):
    f = lambda ts: r*np.sin(ts/r)
    ys = np.add(M0, (f(ts)))
    
    return ys


annuli = annuli_displacements(R, R0, a)

M0 = np.zeros(t)
ts = range(t)
Ms = []

for i in range(len(annuli)):  

    r = annuli[i]
    M = toy_accretion_rate(M0, r, ts)
    Ms.append(M)
    M0 = M

plt.plot(ts, Ms[0])
plt.show()
plt.plot(ts, M)
plt.show()
freq, power = signal.welch(M)
psd = np.multiply(power, freq)
plt.plot(freq, psd)
plt.show()