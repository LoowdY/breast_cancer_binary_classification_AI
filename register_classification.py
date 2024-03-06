#importing all necessary libs/modules

#importing pandas for data import
import pandas as pd
import numpy as np

#importing keras modules and libs
from keras.models import Sequential
from keras.layers import Dense, Dropout


#importing data
pred = pd.read_csv('data\entradas_breast.csv')
classes = pd.read_csv('data\saidas_breast.csv')


#creting neural network and setting parameters
classificator = Sequential()
classificator.add(Dense(units=8, activation='relu', kernel_initializer='normal', input_dim=30))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=8, activation='relu', kernel_initializer='normal'))
classificator.add(Dropout(0.2))
classificator.add(Dense(units=1, activation='sigmoid'))
classificator.compile(optimizer  = 'adam', loss = 'binary_crossentropy',
                       metrics = ['binary_accuracy'])

classificator.fit(pred, classes, batch_size=10, epochs=100)

predict_register = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.78]])

prediction = classificator.predict(predict_register)

print(prediction)

#there a possibility to use a filter like prediction = (prediction >= 0.85). This
#means that the model just outputs True if the result is greater than 0.85.
