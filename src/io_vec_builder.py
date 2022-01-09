import numpy as np
import os

aapl_data = np.load("AAPL_20211210_20220107_data.npy")
fb_data = np.load("FB_20211210_20220107_data.npy")
nvda_data = np.load("NVDA_20211210_20220107_data.npy")

#each sample will be of input_mins + output_min_gap + 1
input_mins = 5 #minutes of data to feed to network
output_min_gap = 1 #minutes between end of input and prediction minute
threshold = 0.1 # how much the output stock price should be greater than the last input price to be considered a win
sample_len = input_mins + output_min_gap + 1

def make_io_vecs(data, stock):
    inputs = [data[i:i+5] for i in range(0, len(data), sample_len)]
    outputs = []
    counts = [0, 0]
    for i in range(6, len(data), sample_len):
        diff = data[i] - data[i-2]
        if diff > threshold:
            outputs.append(1)
            counts[1] += 1
        else:
            outputs.append(0)
            counts[0] += 1
    print(counts)
    if len(inputs) == len(outputs) + 1:
        inputs = inputs[:-1]
    if len(inputs) == len(outputs):
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        idx = np.argsort(np.random.random(outputs.shape[0]))
        inputs = inputs[idx]
        outputs = outputs[idx]
        os.system("rm -f %s_input_data.npy")
        os.system("rm -f %s_output_data.npy")
        np.save("%s_input_data.npy" % stock, inputs)
        np.save("%s_output_data.npy" % stock, outputs)

make_io_vecs(aapl_data, 'AAPL')
make_io_vecs(fb_data, 'FB')
make_io_vecs(nvda_data, 'NVDA')
