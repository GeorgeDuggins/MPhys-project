import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import time
from sklearn.utils import check_random_state
import math

start_time = time.time()
R = 10000 # radius of disk in units of Gravitational radii Rg = GM/c^2
a = 10 # number of annuli
R0 = 1 # inner edge of disc
alpha = 1 # viscocity parameter
H = 1 #disc height
T = 1000000 #number of time values at which we measure Mdot
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
    
def sin_viscoustimescale(r, rmod):
    """
    gives a viscous timescale for the sinusoidal model based on the period of a sin wave
    at the frequency defined by the current position r
    """
    t = (2*np.pi*rmod*r)
    return t

def sin_sample_times(tau, ts, res):
    """
    takes the viscous timescale and calculates the valus of t when accretion rate needs to
    be calculated in order to give "res" readings per viscous timescale
    """
    t = int(round(tau/res))
    times = ts[::t]
    return times 

def sample_times(tau, T, ts, M):
    
    times = np.linspace(0, T-1, len(M), dtype = int)
    return times

def sin_accretion_rate(M0, r, ts, rmod):
    """
    generates a sin wave with amplitude and frequency varying based on displacement and
    gives its product with all the functions at previous r values
    """
    f = lambda ts: (r*np.sin(ts/(rmod*r)))/100000
    m = f(ts)
    M0 = np.take(M0, ts)
    ys = np.multiply(M0, 1+(m))
    
    return ys

def m0accretion_rate(T, beta, tau, r, R):
    
    t = int(T/tau)
    dt = int(tau)
    beta = (1/r)*1000000
    x = generate_lorentzian(t, dt, beta)
    return x
    
def calculate_M(M0, ts, m):
    
    M0 = np.take(M0, ts)
    ys = np.multiply(M0, 1+(m))
    return ys

def generate_lorentzian(N, dt, beta, generate_complex=False, random_state=None):
    """
    This uses the method from Timmer & Koenig [1]_
    Parameters
    ----------
    N : integer
        Number of equal-spaced time steps to generate
    dt : float
        Spacing between time-steps
    beta : float
        Power-law index.  The spectrum will be (1 / f)^beta
    generate_complex : boolean (optional)
        if True, generate a complex time series rather than a real time series
    random_state : None, int, or np.random.RandomState instance (optional)
        random seed or random number generator
    .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
    """
    random_state = check_random_state(random_state)
    dt = float(dt)
    N = int(N)
    
    print(dt, N)

    Npos = int(N / 2)
    Nneg = int((N - 1) / 2)
    domega = (2 * np.pi / dt / N)

    if generate_complex:
        omega = domega * np.fft.ifftshift(np.arange(N) - int(N / 2))
    else:
        omega = domega * np.arange(Npos + 1)

    x_fft = np.zeros(len(omega), dtype=complex)
    x_fft.real[1:] = random_state.normal(0, 1, len(omega) - 1)
    x_fft.imag[1:] = random_state.normal(0, 1, len(omega) - 1)

    fomega = lorentzian(beta*(2*math.pi/1000),omega,(beta/10)*(2*math.pi/1000))
    x_fft[1:] *= fomega[1:]
    x_fft[1:] *= (1./np.sqrt(2))
    
    # by symmetry, the Nyquist frequency is real if x is real
    if (not generate_complex) and (N % 2 == 0):
        x_fft.imag[-1] = 0

    if generate_complex:
        x = np.fft.ifft(x_fft)
    else:
        x = np.fft.irfft(x_fft, N)

    return x

def lorentzian(a, omega, b):
    
    f = lambda omega: (1/b*math.pi)*((b**2)/((a-omega )**2+b**2))
    y = f(omega)
    
    return y

def psd(M):
    freq, power = signal.welch(M)
    psd = np.multiply(power, freq)
    plt.loglog(freq, psd)
    plt.show()

annuli = annuli_displacements(R, R0, a)

M0 = np.ones(T)
ts = range(T)
Ms = []
rmod = 0.2
beta = 100

for i in range(len(annuli)):  

    r = annuli[i]
    tau = sin_viscoustimescale(r, rmod)
    #times = sin_sample_times(tau, ts, res)
    m = m0accretion_rate(T, beta, tau, r, R)
    times = sample_times(tau, T, ts, m)
    M = calculate_M(M0, times, m)
    #M = sin_accretion_rate(M0, r, times, rmod) 
    
    
    M = np.interp(ts, times, M)
    Ms.append(M)
    M0 = M


plt.plot(ts, Ms[0])
plt.show()
plt.plot(ts, M)
plt.show()
psd(M)
print("--- %s seconds ---" % (time.time() - start_time))


