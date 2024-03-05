#importing all necessary libs/modules

#importing pandas for data import
import pandas as pd

#importing train and test module from sklearn
from sklearn.model_selection import train_test_split, cross_val_score

#importing keras modules and libs
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier 


#defining preditors variable
pred = pd.read_csv('data\entradas_breast.csv')

#defining class vriables

classes = pd.read_csv('data\saidas_breast.csv')



def createNetwork():

    #initializing the Neural Network
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
    
    #applying droput technique to the first hidden layer
    classificator.add(Dropout(0.2))
    
    #defining the second hidden layer
    classificator.add(Dense(units = 16,  activation= 'relu', 
                            kernel_initializer= 'random_uniform'))
    
    #applying droput technique to the second hidden layer
    classificator.add(Dropout(0.2))
    
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
    return classificator


#implementing neural network with a classifier
classificator = KerasClassifier(build_fn = createNetwork, epochs = 100, batch_size = 10)

#implementing cross validation
results = cross_val_score(estimator = classificator, X = pred, y = classes, cv = 10, scoring = 'accuracy')

#the CV parameters is 10, so there is 10 results. I choose to apply the mean statistical concept to get the real result. See this below.
real_results = results.mean()

#caculating the stadard deviation (std)
#the bigger std is, the more overfitting is possible.
std = results.std()






