# stock_trend_learner
Neural Network trained on minute-to-minute stock fluctuations to predict the possibility of a quick rise or fall in price, to be used in day-trading applications in either stock or stock options transactions.

**Planning Steps:**
1. Find reliable API to be source of stock information **[yfinance](https://pypi.org/project/yfinance/) works for now**
2. Download a multitude of historical stock data, preferably minute-to-minute prices on stocks exhibiting normal behavior **Wrote a script to download stock prices by the minute for 12/10/21-1/7/22**
3. Build feature vectors of minute-to-minute stock prices, with the price *x* minutes after as the classifier **Experimented with 5 minute stock price "feature vecs" that mapped to a boolean corresponding to if the price of the 7th minute was greater than the 5th by a variable threshold**
4. Train traditional Neural Network on data and see what patterns are found **So far, not much learning is happening, network runs but data provided doesn't have anything useful yet**
5. Change number of output classes to 3: (0: Gain (price of min 15 > price of min 14 by threshold *t*), 1: Loss (price of min 15 < price of min 14 by threshold *t*), 2: Neutral (stock price difference falls between 0 and 1 classifiers) **Done**
6. Convert input data to 14-min line plots and run through convulutional neural network with (2x2) kernel to investigate spatial patterns in the visual price changes, **Built, but still working on gaining useful info from data. See below.**

**So far, the process works as follows:**

[build_data.py](/src/build_data.py)

Using a .txt list of the S&P 100 member's tickers, pull stock prices for every minute in the past 30 days for each ticker with [yfinance](https://pypi.org/project/yfinance/). Then, for each 15 minutes, take the first 14 and save as a 1x14 vector. Take the last minute (15th) and compare it to the 14th. I'm still working on what exactly to use as classifiers, but for the time being I'm using the upper and lower bounds described in Step 5 above to create GAIN, NEUTRAL, and LOSS classes using the 14th and 15th minute data. This gets stored in a 1x1 vector. For each of the 570 samples (30 days = ~22 business days x 6.5 hours per day x 60 min per hour / 15 min per sample = ~570 samples per stock), the index of the 1x14 input vector is correlated with its 1x1 classifier. These get shuffled randomly with indexes still matching and ~570x14 and ~570x1 matrices are returned. This occurs for each of the 100 stocks. At the end, big ~57000x14 and 57000x1 matrices are built with all the data and indexes are then again shuffled randomly but kept maching corresponding data in the two matrices.


[chart_builder.py](/src/chart_builder.py)

Using the large matrix of 1x14 vectors, each of the 14 stock prices are plotted in a [matplotlib](https://matplotlib.org/) plot with some parameters set to make it look like the image below. These plots are then converted to a [numpy](https://numpy.org/) array and are used to create one big ~57000x24x32 matrix of all the plots.

![plot](https://user-images.githubusercontent.com/75876568/149235640-79aa28cb-fc58-4170-835e-b1e43e4aea21.jpg)

[neural_network.py](/src/neural_network.py)

Using the matrices created in the previous two scripts, these are broken into a large training set and a small test set. These are then fed through a Convolutional Neural Network using [Keras](https://keras.io/) with an architecture consisting of two 2D-convolutional layers each with 2x2 kernels, then a maxpooling layer, then two fully connected layers with 1024 and 512 nodes, respectively. I also experimented with  a couple of Dropout layers at one point but they are not included in the current architecture. After the two large fully connected layers, a third connected layer is used to map to the three classifiers using a softmax activation function. The previous layers all use a rectified linear unit activation. The model makes use of [Keras's](https://keras.io/) built-in Adam optimizer and Categorical Cross-Entropy loss function. 

**Results/Progress:** 

So far, I am having difficulties having the network learn the plots. During training, the model quickly overfits the training data and the validation data (I used the testing data for now) drops off until stabilizing at an accuracy signifying the model has learned nothing. Overall, I'm still considering different ways of training the network and possible changes to the architecture, but as of now, the network doesn't tell me much except that day trading is a pretty damn hard job. I guess I knew that starting this project though.

