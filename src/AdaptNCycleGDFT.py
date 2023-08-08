from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from utils.avapi import AlphaVantage as av
from utils.goertzel import Goertzel as G

# from numba import jit
# from timer import Timer

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


price = EPF(close)
