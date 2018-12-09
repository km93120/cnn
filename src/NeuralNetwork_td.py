import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

class NeuralNetwork(object):

    def __init__(self):
        self.model = None
    

    def createModel(self):

       conv1 = keras.layers.Conv2D(filters= 32,kernel_size = 5,strides = 1, padding = "same", #kernel size : 3
                                   data_format = 'channels_last',activation='relu', 
                                   input_shape= (32, 32,3))

       pool1 = keras.layers.MaxPooling2D(pool_size=2,strides = 2,data_format = "channels_last")

       pool1Activation = keras.layers.Activation('relu')

       drop1 = keras.layers.Dropout(0.25)
       
       rnorm1 = keras.layers.BatchNormalization(axis = 3)
 
       conv2  = keras.layers.Conv2D(filters= 32,kernel_size= 3,strides = 1,padding = "same",
                                    data_format = "channels_last",activation='relu');
       
       pool2  = keras.layers.AveragePooling2D(pool_size = 2,strides = 2,data_format = "channels_last")



       rnorm2 = keras.layers.BatchNormalization(axis=3)

       conv3  = keras.layers.Conv2D(filters= 64,kernel_size= 3,strides = 1,padding = "same",#valid
                                    data_format = "channels_last",activation='relu');
       
       pool3  = keras.layers.AveragePooling2D(pool_size = (3,3),strides = 2,data_format = "channels_last")

       drop2 = keras.layers.Dropout(0.5)

       flattening = keras.layers.Flatten(data_format = "channels_last");

       drop3 = keras.layers.Dropout(0.25)

       fc10  = keras.layers.Dense(10);

       softmax = keras.layers.Softmax()
       self.model = keras.Sequential([
                conv1,
                pool1,
                drop1,
               # pool1Activation,
                rnorm1,
                conv2,
                pool2,
                rnorm2,
                conv3,
                pool3,
                drop2,
                flattening,
                drop3,
                fc10,
                softmax,
       ])

        

        
    



        

      

    
        
    
    def train(self, datagen ,train_data, train_labels, epochss):

        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=3,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)
        #optimizer = RMSprop(lr=1, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=keras.optimizers.Adam(lr = 0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        """Train the keras model
        
        Arguments:
            train_data {np.array} -- The training image data
            train_labels {np.array} -- The training labels
            epochs {int} -- The number of epochs to train for
        """

        self.model.fit_generator(datagen.flow(train_data, train_labels, batch_size=32),
                                 steps_per_epoch=len(train_data) / 32,
                                 epochs=epochss,
                                 validation_data=(train_data,train_labels),
                                 #callbacks=[learning_rate_reduction]
                                 )
       # self.model.fit(train_data,train_labels,epochs = epochss,batch_size = 32)

        pass

    def evaluate(self, eval_data, eval_labels):
        """Calculate the accuracy of the model
        
        Arguments:
            eval_data {np.array} -- The evaluation images
            eval_labels {np.array} -- The labels for the evaluation images
        """
        return self.model.evaluate(eval_data, eval_labels, batch_size=32)
        pass

    def test(self, test_data):
        """Make predictions for a list of images and display the results
        
        Arguments:
            test_data {np.array} -- The test images
        """
        predictions = self.model.predict(test_data)
        return predictions
        pass

    ## Exercise 7 Save and load a model using the keras.models API
    def saveModel(self, saveFile="model.h5"):
        """Save a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """

        self.model.save(saveFile);

        pass

    def loadModel(self, saveFile="model.h5"):
        """Load a model using the keras.models API
        
        Keyword Arguments:
            saveFile {str} -- The name of the model file (default: {"model.h5"})
        """
        pass