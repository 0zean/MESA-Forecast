from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from utils.avapi import AlphaVantage as av
from utils.goertzel import Goertzel as G
from utils.timer import Timer

# from numba import jit

api = getcwd() + "/api-key.yaml"

with open(api, "r") as file:
    key = yaml.safe_load(file)

# Retrieve stock data via Alpha Vantage API
data = av(function='TIME_SERIES_INTRADAY',
          symbol='SPY',
          adjusted='true',
          interval='5min',
          extended_hours='false',
          outputsize='full',
          apikey=key['key'])

stock = data.get_extended_data(start_date='2023-05', end_date='2023-08')

print(stock)

close = stock["4. close"]
plt.plot(close.values)
plt.show()

day = '2023-07-05'
bar_count = close.resample('D').count().get(day)
print(bar_count)


# End-point Flattening function
def EPF(data):
    a = data[0]
    b = (data[len(data)-1] - a) / (len(data)-1)

    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i] - (a + b * (i))
    return y


freq_bin = [1/i for i in range(6, 390)]


def rolling_window(data, window, n):
    total = len(data)
    spots = list(range(total-window))
    fpk = []
    epk = []

    for i in range(len(spots)):
        two_pi = 2*np.pi
        newdata = data[spots[i]:(spots[i]+window)]
        
        # Step 1: Endpoint flatten the windowed data
        ff = EPF(newdata)

        # Step 2: For each frequency, calculate goertzel amplitude and phase
        amp_bin, phase_bin = map(list, zip(*[G(ff, f).goertzel() for f in freq_bin]))

        filters = pd.DataFrame({'a': amp_bin, 'f': freq_bin, 'p': phase_bin})

        # Step 3: Filter for the top n frequencies and phases by amplitude
        a_n = filters['a'].nlargest(n=n)
        a_n_idx = list(a_n.index)

        bin = [filters.iloc[idx] for idx in a_n_idx]

        filt = pd.DataFrame({'a': [a[0] for a in bin],
                             'f': [f[1] for f in bin],
                             'p': [p[2] for p in bin]})

        fp = 0
        ep = 0
        # Step 4: fp is the future price and ep is the endpoint price
        for i in range(len(filt)):
            fp += filt['a'][i] * np.cos(two_pi*filt['f'][i]*(window+1) + filt['p'][i])
            ep += filt['a'][i] * np.cos(two_pi*filt['f'][i]*window + filt['p'][i])

        fpk.append(fp)
        epk.append(ep)

    return fpk, epk


t = Timer()
t.start()

fpk, epk = rolling_window(close, 780, 10)

t.stop()

# Step 5: calculate velocity between fp(k) and ep(k) recursively
curve = pd.DataFrame({'fpk': fpk, 'epk': epk})
velocity = curve['fpk'] - curve['epk']

sumv = np.zeros(len(curve))
for k in range(len(curve)):
    sumv[k] = sumv[k-1] + velocity[k]


fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Original + Indicator')
ax1.plot(close[780:].values)
ax2.plot(sumv)
fig.show()
