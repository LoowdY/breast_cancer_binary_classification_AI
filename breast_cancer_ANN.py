#importing all necessary libs/modules

#importing pandas for data import
import pandas as pd

#importing train_test_split module from sklearn to divide train and test dataframes
from sklearn.model_selection import train_test_split

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score, confusion_matrix


#defining preditors variable
pred = pd.read_csv('data\entradas_breast.csv')

#defining class vriables

classes = pd.read_csv('data\saidas_breast.csv')

#dividing train and test dataframes
#using train size as 0.25 (25%), so there is 0.75 (75%) for training
pred_train, pred_test, classes_train, classes_test = train_test_split(pred, classes, test_size = 0.25)

#building the struct for the Neural Network (sequencial)
classificator = Sequential()

#defining a  fully connected Neural Network
'''
Dense function needs a parameter called units;
the units parameter is the number of units is the Numberof columnsinTheDataFrame + neurons in the outputayer)/2;
For this case, there is 30 Columns and 1 neuron in the outputlayers, because its a binary classifier;
the result is 15.5, I choose to use 16. I used relu (rectified Linear Units) as the activation function;
Input_dim is related to the number of elements in the input layer;
'''
#defining the  first hidden layer (between the prameters I define the input (input_dim) layer)
classificator.add(Dense(units = 16,  activation= 'relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30))

#defining the second hidden layer
classificator.add(Dense(units = 16,  activation= 'relu', 
                        kernel_initializer= 'random_uniform'))

#defining the output layer
classificator.add(Dense(units = 1,activation = 'sigmoid'))


#compilation (old, before optimizer1 - below)
#classificator.compile(optimizer  = 'adam', loss = 'binary_crossentropy',
#                      metrics = ['binary_accuracy'])

#initializing the optimizer for the Neral network
#defining the Learning Rate (lr), decay (decay) and the clipvlaue (limitation for the gradient)
#optimizer1 = keras.optimizers.Adam(learning_rate = 0.001, clipvalue = 0.5)

classificator.compile(optimizer  = 'adam', loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])

#visualizing weights
#first hidden layer
weights01 = classificator.layers[0].get_weights()

#second hidden layer
weights02 = classificator.layers[1].get_weights()

#printing weights
print(weights01)
print(weights02)

#SGD (stochastic Gradient Descend) and fit
classificator.fit(pred_train, classes_train, batch_size = 10, epochs = 100)


#predition
predictions = classificator.predict(pred_test)

#making the predictions variable becoming boolean
predictions = (predictions > 0.5)

#implementing metrics for the Neural Network
#OBSERVATION: acc means accurancy and mtx means matrix
acc = accuracy_score(classes_test, predictions)
mtx = confusion_matrix(classes_test, predictions)

#result 

result = classificator.evaluate(pred_test,  classes_test)


