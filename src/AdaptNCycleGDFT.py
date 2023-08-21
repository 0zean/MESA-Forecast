import os
from datetime import datetime
import multiprocessing
import heapq

import fastgoertzel as G
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgoertzel import goertzel as lg

from utils.avapi import AlphaVantage as av
from utils.timer import Timer

api = os.getcwd() + "/api-key.yaml"

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


# Check if stored data is up to date
if os.path.exists("data\data.csv"):
    stock = pd.read_csv("data\data.csv", index_col=0, parse_dates=True)
    print(stock.index[-1])
    
    # if stock.index[-1].strftime("%Y-%m-%D") != datetime.now().strftime("%Y-%m-%D"):
    #     stock = data.get_extended_data(start_date='2023-05', end_date='2023-08')
else:
    stock = data.get_extended_data(start_date='2023-05', end_date='2023-08')


close = stock["4. close"]


# Check how many price 'bars' in a single day
# day = '2023-07-05'
# bar_count = close.resample('D').count().get(day)
# print(bar_count)


# End-point Flattening function
def EPF(data):
    n = len(data)
    a = data[0]
    b = (data[-1] - a) / (n - 1)

    i = np.arange(n)
    y = data - (a + b * i)

    return np.array(y)


freq_bin = np.array([1/i for i in range(6, 391)])
two_pi = 2 * np.pi


def rolling_window(data, window, n):
    two_pi = 2*np.pi
    total = len(data)
    spots = list(range(total-window))
    fpk = []
    epk = []

    for i in range(len(spots)):
        newdata = data[spots[i]:(spots[i]+window)]
        
        # Step 1: Endpoint flatten the windowed data
        ff = EPF(newdata)

        # Step 2: For each frequency, calculate goertzel amplitude and phase
        amp_bin, phase_bin = map(list, zip(*[G.goertzel(ff, f) for f in freq_bin]))

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


def process_spot(i, data, window, n, freq_bin):
    newdata = data[i:(i + window)]
    
    # Step 1: Endpoint flatten the windowed data
    ff = EPF(newdata)

    # Step 2: For each frequency, calculate goertzel amplitude and phase
    amp_bin, phase_bin = zip(*[G.goertzel(ff, f) for f in freq_bin])

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
        fp += filt['a'][i] * np.cos(two_pi * filt['f'][i] * (window + 1) + filt['p'][i])
        ep += filt['a'][i] * np.cos(two_pi * filt['f'][i] * window + filt['p'][i])

    return fp, ep


def process_spot2(i, data, window, n, freq_bin):
    newdata = data[i:(i + window)]

    # Step 1: Endpoint flatten the windowed data
    ff = EPF(newdata)

    # Step 2: For each frequency, calculate goertzel amplitude and phase
    amp_bin, phase_bin = zip(*[G.goertzel(ff, f) for f in freq_bin])

    bin = np.column_stack((amp_bin, freq_bin, phase_bin))

    # Step 3: Filter for the top n frequencies and phases by amplitude
    idx = np.flip(np.argsort(bin[:,0]))[:n]

    top_amp = bin[idx, 0]
    top_freq = bin[idx, 1]
    top_phase = bin[idx, 2]

    # Step 4: Calculate fp and ep using NumPy vectorized operations
    fp = 0
    ep = 0
    for i in range(len(top_amp)):
        fp += top_amp[i] * np.cos(two_pi * top_freq[i] * (window + 1) + top_phase[i])
        ep += top_amp[i] * np.cos(two_pi * top_freq[i] * window + top_phase[i])

    return fp, ep


def parallelized_rolling_window(data, window, n, num_processes=8):
    total = len(data)
    spots = np.arange(total - window)
    fpk = np.zeros(len(spots))
    epk = np.zeros(len(spots))
    
    # Create a pool of worker processes
    with multiprocessing.Pool(num_processes) as pool:
        results = pool.starmap(process_spot2, [(i, data, window, n, freq_bin) for i in spots])

    fpk, epk = zip(*results)

    return fpk, epk


if __name__ == '__main__':
    t = Timer()

    t.start()
    fpk, epk = parallelized_rolling_window(close, 780, 10)
    t.stop()

    # Step 5: calculate velocity between fp(k) and ep(k) recursively
    curve = pd.DataFrame({'fpk': fpk, 'epk': epk})
    velocity = curve['fpk'] - curve['epk']

    sumv = np.zeros(len(curve))
    for k in range(len(curve)):
        sumv[k] = sumv[k-1] + velocity[k]

    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle('Stock + Goertzel Indicator')
    ax1.plot(close[780:].values)
    ax2.plot(sumv)
    plt.show()
