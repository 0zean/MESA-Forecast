from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from alpha_vantage.timeseries import TimeSeries
# from numba import jit

from goertzel import Goertzel as G
# from timer import Timer

api = getcwd() + "/api-key.yaml"

with open(api, "r") as file:
    key = yaml.safe_load(file)

ts = TimeSeries(key=key['key'], output_format="pandas")
data, _ = ts.get_intraday('SPY', interval='5min', outputsize='full')

close = data["4. close"].iloc[::-1]


# End-point Flattening function
def EPF(data):
    a = data[0]
    b = (data[len(data)-1] - a) / (len(data)-1)

    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i] - (a + b * (i))
    return y


price = EPF(close)


# Check to see if Goertzel output = FFT output
def wave(amp, freq, phase, x):
   return amp * np.sin(2*np.pi * freq * x + phase)


x = np.arange(0, 512)
y = wave(1, 1/128, 0, x)
plt.plot(x, y)

amp, phase = G(y, 1/128).goertzel()
print(amp, phase)

amp, phase = G(y, 1/128).goertzelFFT()
print(amp, phase)

amp, phase = G(y, 1/128).goertzelIIR()
print(amp, phase)

tt = np.fft.fft(y)
test = pd.DataFrame()
test['amp'] = np.sqrt(tt.real**2 + tt.imag**2) / (len(y) / 2)
test['freq'] = np.fft.fftfreq(tt.size, d=1)
test['phase'] = np.arctan2(tt.imag, tt.real)

plt.plot(test['freq'], test['amp'])
max_amp = test['amp'].idxmax()
print(test.iloc[max_amp])
