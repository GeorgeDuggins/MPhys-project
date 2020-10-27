import numpy as np
import matplotlib as plt

R = 100 # radius of disk in units of Gravitational radii Rg = GM/c^2
a = 50 # number of annuli
R0 = 1 # inner edge of disc
alpha = 1 # viscocity perameter
H = 1 #disc height
t = 1000 #number of time values at which we measure Mdot

def annuli_displacements(Radius, MinRad, Annuli):
    displacements = np.logspace(np.log10(Radius), np.log10(MinRad), Annuli)
    return displacements

def viscous_frequency(displacements, Radius, Height, alpha):
    f = lambda displacements: displacements**(-3/2)*((Height/Radius)**2)*(alpha/(2*np.pi))
    frequencies = f(displacements)
    return frequencies
    

annuli = annuli_displacements(R, R0, a)

fs = viscous_frequency(annuli, R, H, alpha)
