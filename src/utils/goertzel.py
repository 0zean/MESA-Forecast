import numpy as np
import pandas as pd
from numba import float64, njit
from numba.experimental import jitclass
# from scipy.signal import lfilter


@njit(nogil=True)
def goertzel_optimized(x, f):
    N = len(x)
    k = int(f * N)

    w = 2 * np.pi * k / N
    cw = np.cos(w)
    c = 2 * cw
    sw = np.sin(w)
    z1, z2 = 0.0, 0.0
    
    ip = 0.0
    qp = 0.0
    
    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0
        
    ip = cw * z1 - z2
    qp = sw * z1
    
    amp = np.sqrt(ip**2 + qp**2) / (N / 2)
    phase = np.arctan2(qp, ip)
    
    return amp, phase


spec = [
    ('x', float64[:]),
    ('f', float64),
    ('N', float64),
    ('k', float64)
]


@jitclass(spec)
class Goertzel:
    def __init__(self, x, f):
        self.x = x
        self.f = f
        self.N = len(self.x)
        self.k = int(f * self.N)
        
    # Goertzel algorithm implementation
    def goertzel(self):
        w = 2 * np.pi * self.k / self.N
        cw = np.cos(w)
        c = 2 * cw
        sw = np.sin(w)
        z1, z2 = 0, 0  # Initialization

        for n in range(self.N):
            z0 = self.x[n] + c * z1 - z2
            z2 = z1
            z1 = z0

        ip = cw * z1 - z2
        qp = sw * z1
        
        amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
        phase = np.arctan2(qp, ip)
        return amp, phase

    # Goertzel as the k-th coefficient of an N-point FFT
    # def goertzelFFT(self):
    #     y = np.fft.fft(self.x)
    #     ip = np.real(y[self.k])
    #     qp = np.imag(y[self.k])
        
    #     amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
    #     phase = np.arctan2(qp, ip)
    #     return amp, phase

    # Goertzel as an IIR filter
    def goertzelIIR(self):
        
        def lfilter(b, a, s):
            y = np.zeros_like(s)
            for n in range(len(s)):
                y[n] = np.dot(b, s[n::-1]) - np.dot(a[1:], y[n-1:0:-1])
            return y
        
        W = np.exp(1j * 2 * np.pi * self.k / self.N)
        c = 2 * np.cos(2 * np.pi * self.k / self.N)
        b = [W, -1, 0]  # FIR coefficients
        a = [1, -c, 1]  # IIR coefficients
        y = lfilter(b, a, self.x)
        ip = np.real(y[-1])
        qp = np.imag(y[-1])
        
        amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
        phase = np.arctan2(qp, ip)
        return amp, phase 

if __name__=="__main__":
    # Check to see if Goertzel output = FFT output
    def wave(amp, freq, phase, x):
        return amp * np.sin(2*np.pi * freq * x + phase)

    x = np.arange(0, 512)
    y = wave(1, 1/128, 0, x)

    G = Goertzel(y, 1/128)

    amp, phase = G.goertzel()
    print(f'Goertzel Amp: {amp:.4f}, phase: {phase:.4f}')

    amp, phase = goertzel_optimized(y, 1/128)
    print(f'GoertzelFFT Amp: {amp:.4f}, phase: {phase:.4f}')

    # amp, phase = G.goertzelIIR()
    # print(f'GoertzelIIR Amp: {amp:.4f}, phase: {phase:.4f}')

    ft = np.fft.fft(y)
    FFT = pd.DataFrame()
    FFT['amp'] = np.sqrt(ft.real**2 + ft.imag**2) / (len(y) / 2)
    FFT['freq'] = np.fft.fftfreq(ft.size, d=1)
    FFT['phase'] = np.arctan2(ft.imag, ft.real)

    max_ = FFT.iloc[FFT['amp'].idxmax()]
    print(f'FFT amp: {max_["amp"]:.4f}, '
          f'phase: {max_["phase"]:.4f}, '
          f'freq: {max_["freq"]:.4f}')
