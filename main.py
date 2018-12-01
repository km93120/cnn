from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np


def main():
    #run 1 : 5.7071 loss, 0.1564 acc          .1 epoch : lr = 0.001
    #run 2:  2.3683 loss, 0.1066 acc : 0.1215 .1 epoch : lr = 0.001
    #run 3 : 2.8534     , 0.1001     : 0.1    .1 epoch : lr = 1
    #run 4 : 11.0573    , 0.0772     : 0.0763 .3 epochs: lr = 1 : batch_size = 32
    #run 5 :

    dm = DataManager()
    dm.preprocessData()
    nn = NeuralNetwork()
    nn.createModel()
    nn.train(dm.datagen,dm.train_data,dm.train_labels,10);
    result = nn.evaluate(dm.eval_data,dm.eval_labels)
    print(result[0])
    print(result[1])
    print(nn.test(dm.eval_data));

    

if __name__ == "__main__":
    main()