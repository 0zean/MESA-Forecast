from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from alpha_vantage.timeseries import TimeSeries
from memspectrum import MESA

api = getcwd() + "/api-key.yaml"

with open(api, "r") as file:
    key = yaml.safe_load(file)

ts = TimeSeries(key=key['key'], output_format="pandas")
data, _ = ts.get_intraday('SPY', interval='5min', outputsize='full')


price = data["4. close"].iloc[::-1]

N, dt = len(price), 1

time = np.arange(0, N) * dt

M = MESA()
M.solve(price[:-100])

forecast = M.forecast(price[:-100], length=100, number_of_simulations=1000, include_data=False)
median = np.median(forecast, axis=0)

p5, p95 = np.percentile(forecast, (5, 95), axis=0)

plt.plot(time[:-100], price[:-100], color='k')
plt.fill_between(time[-100:], p5, p95, color='b', alpha=.5, label='90% Cr.')
plt.plot(time[-100:], price[-100:], color='k', linestyle='-.', label='Observed Data')
plt.plot(time[-100:], median, color='r', label='Median Estimate')
plt.show()
