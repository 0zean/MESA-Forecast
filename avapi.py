import requests


def get_data(function, symbol, adjusted, interval,
             extended_hours, month, outputsize, apikey):
    """Retrieves financial time-series data from Alpha Vantage API

    Parameters
    ----------
    function : _str_,
        Type of time-series data e.g., intraday, daily, etc.
    symbol : _str_,
        Stock ticker (in Caps).
    adjusted : _str_,
        Whether or not to use adjusted data (`'true'` or `'false'`).
    interval : _str_,
        time-frame of data e.g., `'1min'`, `'5min'`, `'30min'`.
    extended_hours : _str_,
        Whether or not to include extended hours data (`'true'` or `'false'`).
    month : _str_,
        which month to get data from e.g., `'2023-01'`.
    outputsize : _str_,
        amount of data pulled (`'full'` or `'compact'`).
    apikey : _str_,
        Alpha Vantage API key.

    Returns
    -------
    _dict_,
        JSON dictionary containing time-series data e.g., `Time Series (5min)`.
    -------
    """
    url = (
        f'https://www.alphavantage.co/query?function={function}'
        f'&symbol={symbol}'
        f'&adjusted={adjusted}'
        f'&interval={interval}'
        f'&extended_hours={extended_hours}'
        f'&month={month}'
        f'&outputsize={outputsize}'
        f'&apikey={apikey}'
    )

    r = requests.get(url)
    data = r.json()

    # Keep only the time-series data
    time_series = list(data.keys())[-1]
    data = data[time_series]
    
    return data
