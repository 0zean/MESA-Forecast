from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import avapi as av
from goertzel import Goertzel as G

# from numba import jit
# from timer import Timer

api = getcwd() + "/api-key.yaml"

with open(api, "r") as file:
    key = yaml.safe_load(file)

data = av.get_data(function='TIME_SERIES_INTRADAY',
                   symbol='SPY',
                   adjusted='true',
                   interval='5min',
                   extended_hours='false',
                   month='2023-07',
                   outputsize='full',
                   apikey='RDO0LSA057HUKGIP'
               )

data = pd.DataFrame(data).T
data.index = pd.to_datetime(data.index)
data = data.astype(float)

close = data["4. close"].iloc[::-1]

day = '2023-07-05'
bar_count = close.resample('D').count().get(day)
print(bar_count)

unique_days = close.resample('D').size().shape[0]
print(unique_days)


# End-point Flattening function
def EPF(data):
    a = data[0]
    b = (data[len(data)-1] - a) / (len(data)-1)

    y = np.zeros(len(data))
    for i in range(len(data)):
        y[i] = data[i] - (a + b * (i))
    return y


price = EPF(close)
