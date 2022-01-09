import yfinance as yf
import numpy as np

def get_month(stock, save):

    ticker = yf.Ticker(stock)
    data = np.array(ticker.history(interval='1m', start='2021-12-10', end='2021-12-12'))[:,1] #1 day
    data = np.append(data, np.array(ticker.history(interval='1m', start='2021-12-13', end='2021-12-19'))[:,1]) #5 days
    data = np.append(data, np.array(ticker.history(interval='1m', start='2021-12-20', end='2021-12-26'))[:,1]) #4 days
    data = np.append(data, np.array(ticker.history(interval='1m', start='2021-12-27', end='2022-1-2'))[:,1]) #5 days
    data = np.append(data, np.array(ticker.history(interval='1m', start='2022-1-3', end='2022-1-9'))[:,1]) #5 days
    if save:
        np.save("%s_20211210_20220107_data.npy" % stock, data)
    else:
        return data

get_month('AAPL', True)
get_month('NVDA', True)
get_month('FB', True)
