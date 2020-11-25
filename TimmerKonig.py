import numpy as np
from sklearn.utils import check_random_state
from matplotlib import pyplot as plt
from scipy import signal
import math

def generate_power_law(N, dt, beta, generate_complex=False, random_state=None):
    """Generate a power-law light curve
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
    Returns
    -------
    x : ndarray
        the length-N
    References
    ----------
    .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
    """
    random_state = check_random_state(random_state)
    dt = float(dt)
    N = int(N)

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

    x_fft[1:] *= (1. / omega[1:]) ** (0.5 * beta)
    x_fft[1:] *= (1. / np.sqrt(2))
    
    # by symmetry, the Nyquist frequency is real if x is real
    if (not generate_complex) and (N % 2 == 0):
        x_fft.imag[-1] = 0

    if generate_complex:
        x = np.fft.ifft(x_fft)
    else:
        x = np.fft.irfft(x_fft, N)

    return x

def generate_lorentzian(N, dt, beta, generate_complex=False, random_state=None):
    """Generate a power-law light curve
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
    Returns
    -------
    x : ndarray
        the length-N
    References
    ----------
    .. [1] Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300:707
    """
    random_state = check_random_state(random_state)
    dt = float(dt)
    N = int(N)

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

def lorentzian(a, omega, scale):
    
    f = lambda omega: (1/scale*math.pi)*((scale**2)/((omega-a )**2+scale**2))
    y = f(omega)
    
    return y

def psd(M):
    freq, power = signal.welch(M)
    psd = np.multiply(power, freq)
    plt.loglog(freq, psd)
    plt.show()

T = 1000
dt = 1
beta = 200
t = T/dt

ts = range(0, T, dt)

x2 = generate_lorentzian(t, dt, beta)

plt.plot(ts, x2)
plt.show()
psd(x2)
