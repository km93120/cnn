import numpy as np

class DataManager(object):

    def __init__(self):
        self.train_data = None
        self.train_labels = None
        self.eval_data = None
        self.eval_labels = None

    
    def loadData(self):
        """Load the data from cifar-10-batches. 
           See http://www.cs.toronto.edu/~kriz/cifar.html for instructions on 
           how to do so.
        """
        pass
