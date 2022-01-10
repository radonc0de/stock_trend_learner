import numpy as np
import yfinance as yf
import os
import subprocess

#each sample will be of input_mins + output_min_gap + 1
input_mins = 14 #minutes of data to feed to network
output_min_gap = 1 #minutes between end of input and prediction minute
sample_len = input_mins + output_min_gap

def get_month(stock, date):

    ticker = yf.Ticker(stock)
    #data = np.array(ticker.history(interval='1m', start='2021-12-11', end='2021-12-12'))[:,1] #1 da
    data = np.array(ticker.history(interval='1m', start='2021-12-20', end='2021-12-26'))[:,1]
    data = np.append(data, np.array(ticker.history(interval='1m', start='2021-12-27', end='2022-1-2'))[:,1]) #5 days
    data = np.append(data, np.array(ticker.history(interval='1m', start='2022-1-3', end='2022-1-9'))[:,1]) #5 days
    data = np.append(data, np.array(ticker.history(interval='1m', start='2022-1-10', end='2022-1-11'))[:,1]) #4 days

    make_io_vecs(stock, data, date)

def make_io_vecs(stock, data, date):
    inputs = [data[i:i+input_mins] for i in range(0, len(data), sample_len)]
    outputs = []
    counts = [0 for i in range(11)]
    for i in range(sample_len - 1, len(data), sample_len):
        diff = data[i] - data[i-output_min_gap]
        for ind, j in enumerate([1., 0.5, 0.25, 0.1, 0, -0.1, -0.25, -0.5, -1.]):
            if diff > j:
                outputs.append(ind)
                counts[ind] += 1
                break
            elif j == -1.:
                outputs.append(ind)
                counts[ind] += 1
                break
    print("class distribution:", end="")
    print(counts)
    if len(inputs) == len(outputs) + 1:
        inputs = inputs[:-1]
    if len(inputs) == len(outputs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        idx = np.argsort(np.random.random(outputs.shape[0]))
        inputs = inputs[idx]
        outputs = outputs[idx]
        print(inputs[:10])
        print(outputs[:10])
        os.system("rm -f ./data/%s_%s_input_data.npy" % (stock, date))
        os.system("rm -f ./data/%s_%s_output_data.npy" % (stock, date))
        np.save("./data/%s_%s_input_data.npy" % (stock, date), inputs)
        np.save("./data/%s_%s_output_data.npy" % (stock, date), outputs)

date = subprocess.run(['date', '+%Y-%m-%d'], stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]
print(date)
stocks = ['NVDA', 'FB', 'AAPL', 'MSFT', 'GOOG', 'AMD', 'NKE', 'F']
for stock in stocks: get_month(stock, date)
