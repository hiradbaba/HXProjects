import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree
import sklearn.ensemble
import sklearn.neural_network
import sklearn.svm
import pickle
import torch
import torch.nn as nn
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from models import *
from voting import *

'''Responsible for evaluating data based on given data X,y'''
class Evaluator:
    def __init__(self,X,y):
        self.X = X
        self.y = y
        #loading encoder to handle both numerical values and string values
        with open('nlp_models/encoder.pkl','rb') as f:
            self.le = pickle.load(f)

    '''set new data X and targets y for evaluation'''
    def set_data(self,X,y):
        self.X = X
        self.y = y

    '''Convert targets to the format that is equal to the format of model predictions'''
    def convert_to_proper_y(self,model,y):
        y = self.y
        model_type = model.classes_.dtype
        if model_type == np.int:
            if model_type != self.y.dtype:
                return self.le.transform(y)
        else:
            if model_type != self.y.dtype:
                return self.le.inverse_transform(y)
        return y
    '''Evaluate sklearn models'''
    def evaluate_sklearn(self,path):
        #loads model from the path
        with open(path,'rb') as f:
            model = pickle.load(f)
            #assign a name to model
            name = type(model).__name__
            if isinstance(model,sklearn.neural_network.MLPClassifier):
                name+= " "+ model.solver
            #convert y to the model desired format
            y_tst = self.convert_to_proper_y(model,self.y)
            #plot confusion matrix
            disp2 = sklearn.metrics.plot_confusion_matrix(model,self.X,y_tst,cmap=plt.cm.Blues)
            disp2.ax_.set_title("test confusion matrix " + name)
            #save plot
            plt.savefig(f'figs/test_{name}.png',facecolor='white')
            #plotting the classification report
            plt.figure()
            ax = plt.gca()
            #removing axises from the figure
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #disabling boxes
            plt.box(on=None)
            plt.title(name)
            plt.text(0.5,0.5,sklearn.metrics.classification_report(y_tst,model.predict(self.X)),
                    horizontalalignment='center', verticalalignment='center'
                    )
            plt.savefig(f'figs/{name}_report.png', facecolor='white')

    '''Evaluate torch models'''
    def evaluate_torch(self,path,input_shape,num_features,name):
        #load model from given file
        print(f"evaluate {name}")
        with open(path,'rb') as f:
            #load model from pipeline
            model = pickle.load(f)['net']
            #convert data into tensors
            Xt = torch.from_numpy(self.X).to(torch.float)
            #format data using first num_features to given input shape
            Xt = Xt[:,:num_features].reshape(input_shape)
            #transforming y
            y_tst = self.le.transform(self.y) if self.y.dtype != np.int else self.y

            disp2 = sklearn.metrics.plot_confusion_matrix(model,Xt,y_tst,cmap=plt.cm.Blues)
            disp2.ax_.set_title("test confusion matrix " + name)
            plt.savefig(f'figs/test_{name}.png',facecolor='white')
            
            plt.figure()
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.box(on=None)
            plt.title(name)
            
            y_pred = model.predict(Xt)
            plt.text(0.5,0.5,sklearn.metrics.classification_report(y_tst,y_pred,target_names=self.le.classes_),
                    horizontalalignment='center', verticalalignment='center')
            plt.savefig(f'figs/{name}_report.png', facecolor='white')

    '''Plots the evaluation and report of sklearn models'''
    def show_evaluate_torch(self,path,input_shape,num_features,name):
        print(f"evaluate {name}")
        #same process to evaluate_torch
        with open(path,'rb') as f:
            model = pickle.load(f)['net']

            Xt = torch.from_numpy(self.X).to(torch.float)
            Xt = Xt[:,:num_features].reshape(input_shape)
            
            y_tst = self.le.transform(self.y) if self.y.dtype != np.int else self.y

            disp2 = sklearn.metrics.plot_confusion_matrix(model,Xt,y_tst,cmap=plt.cm.Blues)
            disp2.ax_.set_title("test confusion matrix " + name)
            
            plt.figure()
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.box(on=None)
            plt.title(name)
            
            y_pred = model.predict(Xt)
            plt.text(0.5,0.5,sklearn.metrics.classification_report(y_tst,y_pred,target_names=self.le.classes_),
                    horizontalalignment='center', verticalalignment='center')
            plt.show()

    '''Plots the evaluation and report of torch models'''
    def show_evaluate_sklearn(self,path):
        #same process to evaluate sklearn
        with open(path,'rb') as f:
            model = pickle.load(f)
            name = type(model).__name__
            if isinstance(model,sklearn.neural_network.MLPClassifier):
                name+= " "+ model.solver
            y_tst = self.convert_to_proper_y(model,self.y)
            
            disp2 = sklearn.metrics.plot_confusion_matrix(model,self.X,y_tst,cmap=plt.cm.Blues)
            disp2.ax_.set_title("test confusion matrix" + name )
            #plt.show()

            plt.figure()
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.box(on=None)
            plt.title(name)
            plt.text(0.5,0.5,sklearn.metrics.classification_report(y_tst,model.predict(self.X)),
                    horizontalalignment='center', verticalalignment='center'
                    )
            plt.show()
        
        
            