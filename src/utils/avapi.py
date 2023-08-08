import pandas as pd
import requests
from dateutil.relativedelta import relativedelta


class AlphaVantage():
    """Initializes Alpha Vantage API variables

    Parameters
    ----------
    function : _str_,
        Type of time-series data e.g., `'TIME_SERIES_INTRADAY'`.
    symbol : _str_,
        Stock ticker (in Caps).
    adjusted : _str_,
        Whether or not to use adjusted data (`'true'` or `'false'`).
    interval : _str_,
        time-frame of data e.g., `'1min'`, `'5min'`, `'30min'`.
    extended_hours : _str_,
        Whether or not to include extended hours data (`'true'` or `'false'`).
    outputsize : _str_,
        amount of data pulled (`'full'` or `'compact'`).
    apikey : _str_,
        Alpha Vantage API key.
    """
    def __init__(self, function, symbol, adjusted, interval,
                 extended_hours, outputsize, apikey):
        
        self.function = function
        self.symbol = symbol
        self.adjusted = adjusted
        self.interval = interval
        self.extended_hours = extended_hours
        self.outputsize = outputsize
        self.apikey = apikey
    
    
    def get_data(self, month):
        """Retrieves financial time-series data from Alpha Vantage API

        Parameters
        ----------
        month : _str_,
            which month to get data from e.g., `'2023-01'`.

        Returns
        -------
        _dict_,
            JSON dictionary containing time-series data e.g., `Time Series (5min)`.
        -------
        """
        url = (
            f'https://www.alphavantage.co/query?function={self.function}'
            f'&symbol={self.symbol}'
            f'&adjusted={self.adjusted}'
            f'&interval={self.interval}'
            f'&extended_hours={self.extended_hours}'
            f'&month={month}'
            f'&outputsize={self.outputsize}'
            f'&apikey={self.apikey}'
        )

        r = requests.get(url)
        data = r.json()

        # Keep only the time-series data
        time_series = list(data.keys())[-1]
        data = data[time_series]
                
        return data


    def get_extended_data(self, start_date, end_date):
        """Fetches all data within specified range and 
        combines them into a single dataframe.

        Parameters
        ----------
        start_date : _str_
            Start date of date collection e.g., `'2023-05'`.
        end_date : _str_
            End date of date collection e.g., `'2023-08'`.

        Returns
        -------
        _pandas.DataFrame_
            Chronological dataframe of all data collected.
        -------
        """
        data_frames = []

        current_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        while current_date <= end_date:
            year_month = current_date.strftime('%Y-%m')
            data = self.get_data(month=year_month)

            if data:
                data_df = pd.DataFrame(data).T
                data_df.index = pd.to_datetime(data_df.index)
                data_df = data_df.astype(float)
                data_frames.append(data_df[::-1])
            
            # Move to the next month
            current_date += relativedelta(months=1)

            if current_date > end_date:
                break
        
        if data_frames:
            final_dataframe = pd.concat(data_frames)
            return final_dataframe
        else:
            return None
