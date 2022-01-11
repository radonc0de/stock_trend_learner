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

    return(make_io_vecs(stock, data, date))

def make_io_vecs(stock, data, date):
    inputs = [data[i:i+input_mins] for i in range(0, len(data), sample_len)]
    outputs = []
    counts = [0 for i in range(3)]
    for i in range(sample_len - 1, len(data), sample_len):
        diff = data[i] - data[i-output_min_gap]
        for ind, j in enumerate([0.1, 0, -0.1]):
            if diff > j:
                outputs.append(ind)
                counts[ind] += 1
                break
            elif j == -0.1:
                outputs.append(ind)
                counts[ind] += 1
                break
    print("%s class distribution: %s" % (stock, counts))
    if len(inputs) == len(outputs) + 1:
        inputs = inputs[:-1]
    if len(inputs) == len(outputs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        idx = np.argsort(np.random.random(outputs.shape[0]))
        inputs = inputs[idx]
        outputs = outputs[idx]
        #print(inputs[:10])
        #print(outputs[:10])
        #os.system("rm -f ./data/%s_%s_input_data.npy" % (stock, date))
        #os.system("rm -f ./data/%s_%s_output_data.npy" % (stock, date))
        #np.save("./data/%s_%s_input_data.npy" % (stock, date), inputs)
        #np.save("./data/%s_%s_output_data.npy" % (stock, date), outputs)
        return(inputs, outputs, len(inputs))
    else:
        print("ERROR!!!!!")

with open("./data/sp100.txt") as raw_sp100:
    stocks = raw_sp100.read().split('\n')
    stocks.pop()
date = subprocess.run(['date', '+%Y-%m-%d'], stdout=subprocess.PIPE).stdout.decode('utf-8')[:-1]
print(date)
#stocks = ['NVDA', 'FB', 'AAPL', 'MSFT', 'GOOG', 'AMD', 'NKE', 'F']
result = [get_month(stock, date) for stock in stocks]
inputs = []
outputs = []
length = 0
for i in result:
    inputs.append(i[0])
    outputs.append(i[1])
    length += i[2]

input = np.zeros((length, 14))
output = np.zeros((length))

idx = 0
for j in inputs:
    for k in j:
        input[idx, :] = k
        idx += 1
idx = 0
for j in outputs:
    for k in j:
        output[idx] = k
        idx += 1
print("Shape input: %s, Shape output: %s" % (input.shape, output.shape))

idx = np.argsort(np.random.random(output.shape[0]))
input = input[idx]
output = output[idx]
np.save("./data/input.npy", input)
np.save("./data/output.npy", output)
