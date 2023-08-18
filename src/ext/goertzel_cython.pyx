import numpy as np
cimport numpy as np

cpdef goertzel(np.ndarray[np.double_t, ndim=1] x, double f):
    cdef int N = x.shape[0]
    cdef int k = int(f * N)

    cdef double w = 2.0 * np.pi * k / N
    cdef double cw = np.cos(w)
    cdef double c = 2.0 * cw
    cdef double sw = np.sin(w)
    cdef double z1 = 0.0, z2 = 0.0
    cdef double z0

    cdef int n
    for n in range(N):
        z0 = x[n] + c * z1 - z2
        z2 = z1
        z1 = z0

    cdef double ip = cw * z1 - z2
    cdef double qp = sw * z1

    cdef double amp = np.sqrt(ip**2 + qp**2) / (N / 2)
    cdef double phase = np.arctan2(qp, ip)
    
    return amp, phase
