import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import numpy as np
x = []
y = []
nvda_i = np.load("NVDA_input_data.npy")
aapl_i = np.load("AAPL_input_data.npy")
nvda_o = np.load("NVDA_output_data.npy")
aapl_o = np.load("AAPL_output_data.npy")

batch_size = 16
num_classes = 2
epochs = 50
nsamp = (5, 1)

N = 2000 #seperator for train/test
x_train = nvda_i
y_train = nvda_o
x_test = aapl_i
y_test = aapl_o

#change output vecs to use one-hot notation
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=nsamp))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'])
model.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test,y_test))
