# stock_trend_learner
Neural Network trained on minute-to-minute stock fluctuations to predict the possibility of a quick rise or fall in price, to be used in day-trading applications in either stock or stock options transactions.

Stages:
1. Find reliable API to be source of stock information
2. Download a multitude of historical stock data, preferably minute-to-minute prices on stocks exhibiting normal behavior
3. Build feature vectors of minute-to-minute stock prices, with the price X minutes after as the classifier
4. Train traditional Neural Network on data and see what patterns are found
