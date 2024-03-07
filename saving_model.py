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


#outputting, in json format, the hiperparameters of the DL model
json_classificator = classificator.to_json()

#saving in the current directory the json file (hiperparameters)

with open('json_breast_classificator.json', 'w') as json_file:
    json_file.write(json_classificator)
    
#saving the weights configs
classificator.save_weights('weights_breast_cancer_classificator.h5')
