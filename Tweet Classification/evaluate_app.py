import evaluator
import numpy as np
import pandas as pd

'''If there is a gap between training and evaluation using data/last_*.csv 
    gives the oppurtunity to store the latest dataset in order to evaluate
    model on the dataset which the model was trained on.
'''

#Loads test data from last_test
data = pd.read_csv('data/last_test.csv')
X_test,y_test = data.values[:,:-1].astype('float'), data.values[:,-1].astype('int')
#Create Evaluator and evaluate all models
evaluator = evaluator.Evaluator(X_test,y_test)
evaluator.evaluate_sklearn('models/DT.pkl')
evaluator.evaluate_sklearn('models/adaboost.pkl')
evaluator.evaluate_sklearn('models/MLPClassifier.pkl')
evaluator.evaluate_sklearn('models/svm.pkl')
evaluator.evaluate_sklearn('models/RF.pkl')

evaluator.evaluate_sklearn('models/vote_clf3.pkl')

evaluator.evaluate_torch('models/torch_conv1d.pkl',(-1,1,50),50,'Conv1D')
evaluator.evaluate_torch('models/torch_conv2d.pkl',(-1,1,7,7),49,'Conv2D')
evaluator.evaluate_torch('models/torch_lstm.pkl',(-1,25,2),50,"LSTM")
evaluator.evaluate_torch('models/torch_rnn.pkl',(-1,25,2),50,"RNN")
evaluator.evaluate_torch('models/torch_rcnn.pkl',(-1,1,5,10),50,'LSTM-CNN')