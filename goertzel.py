import numpy as np
from scipy.signal import lfilter


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
    def goertzelFFT(self):
        y = np.fft.fft(self.x)
        ip = np.real(y[self.k])
        qp = np.imag(y[self.k])
        
        amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
        phase = np.arctan2(qp, ip)
        return amp, phase
    
    # Goertzel as an IIR filter
    def goertzelIIR(self):
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
