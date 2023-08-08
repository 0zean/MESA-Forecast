import numpy as np
import pandas as pd
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

if __name__=="__main__":
    # Check to see if Goertzel output = FFT output
    def wave(amp, freq, phase, x):
        return amp * np.sin(2*np.pi * freq * x + phase)

    x = np.arange(0, 512)
    y = wave(1, 1/128, 0, x)

    amp, phase = Goertzel(y, 1/128).goertzel()
    print(f'Goertzel Amp: {amp:.4f}, phase{phase:.4f}')

    amp, phase = Goertzel(y, 1/128).goertzelFFT()
    print(f'GoertzelFFT Amp: {amp:.4f}, phase{phase:.4f}')

    amp, phase = Goertzel(y, 1/128).goertzelIIR()
    print(f'GoertzelIIR Amp: {amp:.4f}, phase{phase:.4f}')

    tt = np.fft.fft(y)
    test = pd.DataFrame()
    test['amp'] = np.sqrt(tt.real**2 + tt.imag**2) / (len(y) / 2)
    test['freq'] = np.fft.fftfreq(tt.size, d=1)
    test['phase'] = np.arctan2(tt.imag, tt.real)

    max_ = test.iloc[test['amp'].idxmax()]
    print(f'FFT amp: {max_["amp"]:.4f}, '
          f'phase: {max_["phase"]:.4f}, '
          f'freq: {max_["freq"]:.4f}')
