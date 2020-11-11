import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import time

start_time = time.time()
R = 10000 # radius of disk in units of Gravitational radii Rg = GM/c^2
a = 10 # number of annuli
R0 = 1 # inner edge of disc
alpha = 1 # viscocity parameter
H = 1 #disc height
t = 1000000 #number of time values at which we measure Mdot
res = 50 #number of values calculated per timescale


def annuli_displacements(Radius, MinRad, Annuli):
    """
    Uses the radius and inner radius of the disc to generate the positions along "r" of a 
    specified number of logarithmically spaced annuli.
    """
    displacements = np.logspace(np.log10(Radius), np.log10(MinRad), Annuli)
    return displacements

def viscous_frequency(displacements, Radius, Height, alpha):
    """
    Viscous frequency formula from A&U
    """
    f = lambda displacements: displacements**(-3/2)*((Height/Radius)**2)*(alpha/(2*np.pi))
    frequencies = f(displacements)
    return frequencies
    
def toy_accretion_rate(M0, r, ts):
    """
    generates a sin wave with amplitude and frequency varying based on displacement and
    gives its product with all the functions at previous r values
    """
    f = lambda ts: (r*np.sin(ts/(rmod*r)))/100000
    m = f(ts)
    ys = np.multiply(M0, 1+(m))
    
    return ys


annuli = annuli_displacements(R, R0, a)

M0 = np.ones(t)
ts = range(t)
Ms = []
rmod = 4

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
plt.loglog(freq, psd)
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))