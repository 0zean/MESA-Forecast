from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta

import utils.avapi as av
from utils.goertzel import Goertzel as G

# from numba import jit
# from timer import Timer

api = getcwd() + "/api-key.yaml"

with open(api, "r") as file:
    key = yaml.safe_load(file)


def get_data_range(start_date, end_date):
    data_frames = []  # List to store individual dataframes for each month

    current_date = start_date
    while current_date <= end_date:
        year_month = current_date.strftime('%Y-%m')
        data = av.get_data(function='TIME_SERIES_INTRADAY',
                   symbol='SPY',
                   adjusted='true',
                   interval='5min',
                   extended_hours='false',
                   month=year_month,
                   outputsize='full',
                   apikey=key['key']
               )
        if data:
            data_df = pd.DataFrame(data).T
            data_df.index = pd.to_datetime(data_df.index)
            data_df = data_df.astype(float)
            data_frames.append(data_df[::-1])
        
        current_date += relativedelta(months=1)  # Move to the next month

        if current_date > end_date:
            break
    
    if data_frames:
        final_dataframe = pd.concat(data_frames)
        return final_dataframe
    else:
        return None  # No data retrieved

s = pd.Timestamp('2023-05')
e = pd.Timestamp('2023-08')

data = get_data_range(start_date=s, end_date=e)

print(data)


close = data["4. close"]

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
