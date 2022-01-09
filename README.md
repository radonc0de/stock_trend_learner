# stock_trend_learner
Neural Network trained on minute-to-minute stock fluctuations to predict the possibility of a quick rise or fall in price, to be used in day-trading applications in either stock or stock options transactions.

Steps:
1. Find reliable API to be source of stock information **[yfinance](https://pypi.org/project/yfinance/) works for now**
2. Download a multitude of historical stock data, preferably minute-to-minute prices on stocks exhibiting normal behavior **Wrote a script to download stock prices by the minute for 12/10/21-1/7/22**
3. Build feature vectors of minute-to-minute stock prices, with the price *x* minutes after as the classifier **Experimented with 5 minute stock price "feature vecs" that mapped to a boolean corresponding to if the price of the 7th minute was greater than the 5th by a variable threshold**
4. Train traditional Neural Network on data and see what patterns are found **So far, not much learning is happening, network runs but data provided doesn't have anything useful yet**
5. Convert input data to 14-min line plots and run through convulutional neural network with (2x2) kernel to investigate spatial patterns in the visual price changes, change number of output classes to 3: (0: Gain (price of min 15 > price of min 14 by threshold *t*), 1: Loss (price of min 15 < price of min 14 by threshold *t*), 2: Neutral (stock price difference falls between  0 and 1 classifiers)
