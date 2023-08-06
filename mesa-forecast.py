from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from alpha_vantage.timeseries import TimeSeries
from memspectrum import MESA
from sklearn.linear_model import LinearRegression

api = getcwd() + "/api-key.yaml"

with open(api, "r") as file:
    key = yaml.safe_load(file)

ts = TimeSeries(key=key['key'], output_format="pandas")
data, _ = ts.get_intraday('SPY', interval='5min', outputsize='full')

price = data["4. close"].iloc[::-1]


# End-point Flattening function
def EPF(data):
    a = data[0]
    b = (data[len(data)-1] - a) / (len(data)-1)

    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i] - (a + b * (i))
    return y


price = EPF(price)

N, dt = len(price), 1

time = np.arange(0, N) * dt

M = MESA()
h = -100
M.solve(price[:h])

forecast = M.forecast(price[:h], length=100, number_of_simulations=1000, include_data=False)
median = np.median(forecast, axis=0)

p5, p95 = np.percentile(forecast, (5, 95), axis=0)

plt.plot(time[:h], price[:h], color='k')
plt.fill_between(time[h:], p5, p95, facecolor='turquoise',alpha=0.5, label='90% Cr.')
plt.plot(time[h:], price[h:], color='k', linestyle='-.', label='Observed Data')
plt.plot(time[h:], median, color='r', label='Median Estimate')
plt.show()
