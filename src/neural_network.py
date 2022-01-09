import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
import numpy as np

x_train = np.load("./data/GOOG_2022-01-09_input_data.npy")
x_test = np.load("./data/NVDA_2022-01-09_input_data.npy")
y_train = np.load("./data/GOOG_2022-01-09_output_data.npy")
y_test = np.load("./data/NVDA_2022-01-09_output_data.npy")

batch_size = 16
num_classes = 9
epochs = 50
nsamp = (5, 1)

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
