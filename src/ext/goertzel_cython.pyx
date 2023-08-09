import numpy as np
cimport numpy as np
from scipy.signal import lfilter

cdef class Goertzel:
    cdef double[:] _x
    cdef public int N
    cdef public int k

    def __init__(self, np.ndarray[np.double_t, ndim=1] x, double f):
        self._x = x
        self.N = len(self.x)
        self.k = int(f * self.N)


    property x:
        def __get__(self):
            return self._x

    cpdef goertzel(self):
        cdef int n
        cdef double w = 2 * np.pi * self.k / self.N
        cdef double cw = np.cos(w)
        cdef double c = 2 * cw
        cdef double sw = np.sin(w)
        cdef double z1 = 0, z2 = 0
        cdef double z0

        for n in range(self.N):
            z0 = self._x[n] + c * z1 - z2
            z2 = z1
            z1 = z0

        cdef double ip = cw * z1 - z2
        cdef double qp = sw * z1

        cdef double amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
        cdef double phase = np.arctan2(qp, ip)
        return amp, phase

    cpdef goertzelFFT(self):
        cdef np.ndarray[complex, ndim=1] y = np.fft.fft(self.x)
        cdef double ip = y[self.k].real
        cdef double qp = y[self.k].imag

        cdef double amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
        cdef double phase = np.arctan2(qp, ip)
        return amp, phase

    cpdef goertzelIIR(self):
        cdef int i
        cdef double W = np.exp(1j * 2 * np.pi * self.k / self.N)
        cdef double c = 2 * np.cos(2 * np.pi * self.k / self.N)
        cdef double[3] b = [W, -1, 0]
        cdef double[3] a = [1, -c, 1]
        cdef np.ndarray[np.double_t, ndim=1] y = lfilter(b, a, self.x)
        cdef double ip = y[-1]
        cdef double qp = 0.0

        cdef double amp = np.sqrt(ip**2 + qp**2) / (self.N / 2)
        cdef double phase = np.arctan2(qp, ip)
        return amp, phase
