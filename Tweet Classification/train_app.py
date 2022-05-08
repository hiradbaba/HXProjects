from trainer import Trainer,SkorchTrainer,VotingTrainer
from models import *
import data_process
import pandas as pd
import numpy as np
import pickle
import sklearn.tree
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
'''This script is to train our models'''

processor = data_process.Preprocessor('data/tweet_data.csv')
#make a training and testing set with preprocessor class
X_train,X_test,y_train,y_test = processor.create_training_test_set('transform_data','encoder')
#sklearn trainer
trainer = Trainer(X_train,y_train)


# #with sample params,in order to observe the actual learning process used in the project set params to None
trainer.train_decision_tree(params = {'splitter':['random']},cv=2)
trainer.train_adaboost(params = {'learning_rate':[0.01]},cv=2)
trainer.train_mlp_classifer(params = {'solver': ['sgd']},cv=2)
trainer.train_random_forest(params = {'criterion':["entropy"]},cv=2)
trainer.train_svm_poly(params = {'degree':[2]},cv=2)
trainer.train_svm_rbf(params = {'kernel':['rbf']},cv=2)
#reports the results of every trained model located in trainer dictionary
for key in trainer.model_selectors:
    print(trainer.get_report(key))

#Skorch trainer
torch_trainer = SkorchTrainer(X_train,y_train)
torch_trainer.train_conv1d(Conv1dText,50)
#converts vectors into 7x7 matrices
torch_trainer.train_conv2d(CBR_Network,49,(7,7))
input_size = 2
torch_trainer.train_rnn(RecurrentNN,params={
    'input_size': input_size,
    'seq_length': X_train.shape[1]//input_size,
    'hidden_size': 15,
    'num_layers': 3,
    'num_classes': 3
})
torch_trainer.train_lstm(LSTM,params={
    'input_size': input_size,
    'seq_length': X_train.shape[1]//input_size,
    'hidden_size': 3,
    'num_layers': 2,
    'num_classes': 3
})

torch_trainer.train_rcnn(ConvRNN,params={
    'hidden_size': 30,
    'out_channels': 30,
    'shape':(5,10)
})
#loading models for ensembling in voting classifier
with open('models/adaboost.pkl','rb') as f1:
    adaboost_clf = pickle.load(f1)

with open('models/DT.pkl','rb') as f:
    dt_clf = pickle.load(f)
with open('models/MLPClassifier.pkl','rb') as f2:
    mlp_clf = pickle.load(f2)
with open('models/svm.pkl','rb') as f3:
    svm_clf = pickle.load(f3)

models = [('adaboost',adaboost_clf),('mlp',mlp_clf),('dt',dt_clf),('svm',svm_clf)]  
voting_trainer = VotingTrainer(X_train,y_train)
voting_trainer.train_vote_clf(models,10,10)
print(voting_trainer.get_report())
