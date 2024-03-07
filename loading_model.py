#importing important packages
import pandas as pd
import numpy as np
from keras.models import model_from_json


#importing data and defining preditors and classes variables
pred = pd.read_csv('data\entradas_breast.csv')
classes = pd.read_csv('data\saidas_breast.csv')


#opening the file
file = open('json_breast_classificator.json','r')
network_str = file.read()

#closing the file
file.close()

#inilializing the model from the file configuration
classificator = model_from_json(network_str)

#inilializing the model  weights from the file configuration
classificator.load_weights('weights_breast_cancer_classificator.h5')

#classification data
predict_register = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.78]])


#predicting/classificating the register
prediction = classificator.predict(predict_register)

#printing the classification
print(prediction)

#compiling the metrics for the neural network model
classificator.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])


#the final result variable evaluates, in this order, the loss function and the accuracy of the model.
final_result = classificator.evaluate(pred, classes)
