import model_optimizer
import numpy as np
import sklearn
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import pickle
import sklearn.svm,sklearn.tree,sklearn.ensemble,sklearn.neural_network
from voting import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from skorch import NeuralNet, callbacks, NeuralNetClassifier
from sklearn.pipeline import Pipeline
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

'''Responsible for training sklearn models given data [X] and targets [y]'''
class Trainer:
    def __init__(self,X_train,y_train):
        self.X = X_train
        self.y = y_train
        # stores the latest trained models in order to reuse
        self.model_selectors = {
            'adaboost': None,
            'dt': None,
            'svm_rbf': None,
            'svm_poly': None,
            'mlp': None,
            'rf': None,
        }

    #train adaboost model with defualt or given parameters (params=None is the process used to train the models for this project)
    def train_adaboost(self, name='adaboost2',random=False,cv=5,n_jobs=-1, params=None):
        print("[Adaboost] Hyperparameter Search")
        #define model
        adaboost_clf = sklearn.ensemble.AdaBoostClassifier(random_state=0)
        #define model selector
        model_selector = model_optimizer.ModelSelector(adaboost_clf,self.X, self.y)
        #load default params
        if params is None:
            estimators_range = np.arange(500,1001,100)
            lr_range = np.logspace(-2,2,4)
            algorithms = ['SAMME','SAMME.R']
            base_estimator = [ sklearn.tree.DecisionTreeClassifier(max_depth=i,random_state=0) for i in range(1,4)
                            ]
            params = {'n_estimators':estimators_range,
                        'learning_rate':[0.01],
                        'algorithm':algorithms,
                        'base_estimator': base_estimator
                        }  
        #start grid search
        search_cv = model_selector.parameter_search(params,random,cv=cv,n_jobs=n_jobs)
        with open(f'models/{name}.pkl','wb') as file:
            #store resulting model in file
            pickle.dump(model_selector.best_estimator,file)
        #stor resulting model in class dictionary
        self.model_selectors['adaboost'] = model_selector
        print("[Result] ", search_cv.best_estimator_)

    #train mlp classifier model with defualt or given parameters (params=None is the process used to train the models for this project)
    def train_mlp_classifer(self, name='MLPClassifer2',random=False,cv=5,n_jobs=-1, params=None):
        print("[MLP] Hyperparameter Search")
        #same process as adaboost
        mlp_clf = sklearn.neural_network.MLPClassifier(random_state=0)
        model_selector = model_optimizer.ModelSelector(mlp_clf,self.X,self.y)
        if params is None:
            params = {
            'hidden_layer_sizes': [(25,12,6),(25,10),(25,),(60,30,10),
                                    (50,50,25,25),(25,10,5),(50,10,3),(40,25,10,6)],
            'activation': ['tanh', 'relu','logistic'],
            'solver': ['sgd', 'adam','lbfgs'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant','adaptive'],
            'max_iter':[500,1000]
                    }

        search_cv = model_selector.parameter_search(params,random,cv=cv,n_jobs=n_jobs)
        with open(f'models/{name}.pkl','wb') as file:
            pickle.dump(model_selector.best_estimator,file)
        self.model_selectors['mlp'] = model_selector
        print("[Result] ", search_cv.best_estimator_)

    #train random forest classifier model with defualt or given parameters (params=None is the process used to train the models for this project)
    def train_random_forest(self, name='rf2',random=False,cv=5,n_jobs=-1, params=None):
        print("[RF] Hyperparameter Search")
        #same process as adaboost
        random_forest = sklearn.ensemble.RandomForestClassifier(random_state=0)
        model_selector = model_optimizer.ModelSelector(random_forest,self.X,self.y)
        if params is None:
            params = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 20, num = 5)],
               'max_features': ['auto'],
               'max_depth': [int(x) for x in np.linspace(2, 30, num = 10)],
                'min_samples_split': [2, 5],
               'bootstrap': [True, False],
              'criterion':['gini',"entropy"]
              }
        search_cv = model_selector.parameter_search(params,random,cv=cv,n_jobs=n_jobs)
        with open(f'models/{name}.pkl','wb') as file:
            pickle.dump(model_selector.best_estimator,file)
        self.model_selectors['rf'] = model_selector
        print("[Result] ", search_cv.best_estimator_)
    #train decision tree classifier model with defualt or given parameters (params=None is the process used to train the models for this project)
    def train_decision_tree(self, name='DT2',random=False,cv=5,n_jobs=-1, params=None):
        print("[DT] Hyperparameter Search")
        #same process as adaboost
        DT = sklearn.tree.DecisionTreeClassifier(random_state=0)
        model_selector = model_optimizer.ModelSelector(DT,self.X,self.y)
        if params is None:
            params = {
                    'criterion':['gini', 'entropy'],
                    'splitter':['best', 'random'],
                    'max_depth':np.hstack((np.arange(2, 15, 1), np.arange(15, 30, 2), np.arange(30, 100, 5))),
                    'min_samples_split':np.arange(2, 10, 1),
                    'max_features':['auto', 'sqrt', 'log2'],
                    }
        search_cv = model_selector.parameter_search(params,random,cv=cv,n_jobs=n_jobs)
        with open(f'models/{name}.pkl','wb') as file:
            pickle.dump(model_selector.best_estimator,file)
        self.model_selectors['dt'] = model_selector
        print("[Result] ", search_cv.best_estimator_)
    #train svm with rbf kernel  with defualt or given parameters (params=None is the process used to train the models for this project)
    def train_svm_rbf(self, name='svm_rbf2',random=False,cv=5,n_jobs=-1, params=None):
        print("[SVM-RBF] Hyperparameter Search")
        #same process as adaboost
        svm = sklearn.svm.SVC(random_state = 0)
        model_selector = model_optimizer.ModelSelector(svm,self.X,self.y)
        if params is None:
            df_shape = ['ovo']
            c = np.logspace(-1, 2, 4)
            params = [{'kernel':['rbf'], 
                            'gamma':np.logspace(-2, 1, 4), 
                            'C':c, 
                            'decision_function_shape':df_shape}]
        search_cv = model_selector.parameter_search(params,random,cv=cv,n_jobs=n_jobs)
        with open(f'models/{name}.pkl','wb') as file:
            pickle.dump(model_selector.best_estimator,file)
        self.model_selectors['svm_rbf'] = model_selector
        print("[Result] ", search_cv.best_estimator_)
    #train svm with poly kernel classifier model with defualt or given parameters (params=None is the process used to train the models for this project)
    def train_svm_poly(self, name='svm_poly2',random=False,cv=5,n_jobs=-1, params=None):
        print("[SVM-POLY] Hyperparameter Search")
        #same process as adaboost
        svm = sklearn.svm.SVC(random_state = 0)
        model_selector = model_optimizer.ModelSelector(svm,self.X,self.y)
        if params is None:
            df_shape = ['ovo']
            c = np.logspace(-1, 2, 4)
            params = [{'kernel':['poly'],
                         'C':c,
                         'degree':[3, 4],
                         'coef0':np.logspace(0, 2, 3), 
                         'decision_function_shape':df_shape}]

        search_cv = model_selector.parameter_search(params,random,cv=cv,n_jobs=n_jobs)
        with open(f'models/{name}.pkl','wb') as file:
            pickle.dump(model_selector.best_estimator,file)
        self.model_selectors['svm_poly'] = model_selector
        print("[Result] ", search_cv.best_estimator_)

    #returns classification report of model dictionary with key = [key]
    def get_report(self,key):
        y_pred = self.model_selectors[key].best_estimator.predict(self.X)
        return self.model_selectors[key].best_report(self.y,y_pred)

'''Responsible for training the voti classifer on given data[X] and targets [y]'''
class VotingTrainer:
    def __init__(self,X_train,y_train):
        self.X = X_train
        self.y = y_train
        self.clf = None
        self.best_weight = None
        #loads encoder for handling prediction differences
        with open('nlp_models/encoder.pkl','rb') as f:
            self.le = pickle.load(f)
        if self.y.dtype == 'int' or self.y.dtype == 'float':
                self.y = self.le.inverse_transform(self.y.astype('int'))

    # makes a random matrix with shape (row,col)
    def make_random_weight(self,row,col):
        np.random.seed(0)
        x = np.random.rand(row,col)
        return x
    #computes the accuracy of a voting classifier on data X and targets y
    def compute_acc_score(self,voting_clf,X,y):
            
            return sklearn.metrics.accuracy_score(
                                y
                                ,voting_clf.predict(X))
       
    #Computes accuracy of voting classifiers with random weights and returns the weights and accuracy scores
    def make_weight_dataset(self,models,X,y,size):
        print('Making random weight matrix')
        W = self.make_random_weight(size,len(models))
        res = []
        for w in W:

            res.append(self.compute_acc_score(VotingClassifier(estimators = models,voting='hard',weights=w),X,y))
        return W,res
    #computes score of voting classifiers for a fraction of data with [max_iter] random vectors
    def train_vote_clf(self,models,end_point=100,max_iter = 2000):
        W,yw = self.make_weight_dataset(models,self.X[:end_point]
                                            ,self.y[:end_point],max_iter)
        #gets the best weight
        best_w = W[yw.index(max(yw))] 
        self.best_weight = best_w
        self.clf = VotingClassifier(estimators = models,voting='hard',weights=best_w)
        with open("models/vote_clf3.pkl",'wb') as f:
            pickle.dump(self.clf,f)

    def get_vote_clf(self):
        return self.clf
    
    def get_report(self):
        return classification_report(self.y,self.clf.predict(self.X))
    
    def get_test_report(self,X,y):
        return classification_report(y,self.clf.predict(X))

'''Responsible for training torch models'''
class SkorchTrainer:
    def __init__(self,X_train,y_train):
        with open('nlp_models/encoder.pkl','rb') as f:
            self.le = pickle.load(f)
        #convert data to tensors
        self.X = torch.from_numpy(X_train)
        self.y = torch.from_numpy(self.le.transform(y_train)) if y_train.dtype != 'int' else torch.from_numpy(y_train)
        #same as trainer
        self.models ={
            'conv1d': None,
            'conv2d': None,
            'RNN': None,
            'LSTM': None,
            'RCNN': None
        }
    #loads data tensors from data loader
    def get_tensors(self):
        loader = DataLoader(TensorDataset(self.X,self.y),len(self.X),shuffle=True)
        return next(iter(loader))

    #trains 1d convolution
    def train_conv1d(self,class_instance,length):
        print("Start training CONV1D")
        torch.manual_seed(0)
        #uses cuda if available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = NeuralNetClassifier(
            module = class_instance,
            device = device,
            criterion = nn.CrossEntropyLoss,
            optimizer = torch.optim.SGD,
            lr = 0.05, #0.05
            optimizer__momentum=0.9, #0.9
            #prints info while training
            callbacks=[
                ('tr_acc', callbacks.EpochScoring(
                    'accuracy',
                    lower_is_better=False,
                    on_train=True,
                    name='train_acc'
                    
                )),
            ],
            batch_size = 128,#128
            max_epochs= 12 #12
        )
        #reshaping tensors
        Xt, yt = self.get_tensors()
        Xt = Xt.reshape(-1,1,length)
        #fitting
        model.fit(Xt.to(torch.float),yt.to(torch.long))
        #saving model
        model_save = Pipeline([
            ('net', model),
        ])
        with open('models/torch_conv1d.pkl','wb') as f:
            pickle.dump(model_save,f)
        
        self.models['conv1d'] = model
    
    def train_conv2d(self,class_instance,num_features,shape):
        print("Start training CONV2D")
        #same as conv1d
        torch.manual_seed(0)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = NeuralNetClassifier(
            module = class_instance,
            device = device,
            criterion = nn.CrossEntropyLoss,
            optimizer = torch.optim.SGD,
            lr = 0.01,
            optimizer__momentum=0.9,
            callbacks=[
                ('tr_acc', callbacks.EpochScoring(
                    'accuracy',
                    lower_is_better=False,
                    on_train=True,
                    name='train_acc'
                    
                )),
            ],
            batch_size = 128,
            max_epochs= 10 
        )
        Xt, yt = self.get_tensors()
        X_2d = Xt[:,:num_features]
        X_2d = X_2d.reshape(-1,1,shape[0],shape[1])
        model.fit(X_2d.to(torch.float),yt.to(dtype=torch.long))
        model_save = Pipeline([
            ('net', model),
        ])
        with open('models/torch_conv2d.pkl','wb') as f:
            pickle.dump(model_save,f)
        
        self.models['conv2d'] = model

    def train_rnn(self,class_instance,params):
        #same as conv1d
        print("Start training RNN")
        torch.manual_seed(0)
        model = NeuralNetClassifier(module = class_instance,
                                    module__input_size = params['input_size'],
                                    module__hidden_size = params['hidden_size'],
                                    module__num_layers = params['num_layers'],
                                    module__seq_length  = params['seq_length'],
                                    module__num_classes = params['num_classes'],
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                    criterion = nn.CrossEntropyLoss,
                                    optimizer = torch.optim.SGD,
                                    lr = 0.008,
                                    optimizer__momentum=0.9,
                                    callbacks=[('tr_acc', callbacks.EpochScoring('accuracy',
                                                                                lower_is_better=False,
                                                                                on_train=True,
                                                                                name='train_acc'))],
                                    batch_size = 2000,
                                    max_epochs= 40)
        Xt,yt = self.get_tensors()
        Xtt = Xt.reshape(-1,params['seq_length'],params['input_size'])
        model.fit(Xtt.to(torch.float),yt.to(dtype=torch.long))
        model_save = Pipeline([
            ('net', model),
        ])
        with open('models/torch_rnn.pkl','wb') as f:
            pickle.dump(model_save,f)
        
        self.models['RNN'] = model
    
    def train_lstm(self,class_instance,params):
        #same as conv1d
        print("Start training LSTM")
        model = NeuralNetClassifier(module = class_instance,
                                    module__input_size = params['input_size'],
                                    module__hidden_size = params['hidden_size'],
                                    module__num_layers = params['num_layers'],
                                    module__seq_length  = params['seq_length'],
                                    module__num_classes = params['num_classes'],
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                    criterion = nn.CrossEntropyLoss,
                                    optimizer = torch.optim.Adam,
                                    lr = 0.008,
                                    callbacks=[('tr_acc', callbacks.EpochScoring('accuracy',
                                                                                lower_is_better=False,
                                                                                on_train=True,
                                                                                name='train_acc'))],
                                    batch_size = 2000,
                                    max_epochs= 70)#70
        Xt,yt = self.get_tensors()
        Xtt = Xt.reshape(-1,params['seq_length'],params['input_size'])
        model.fit(Xtt.to(torch.float),yt.to(dtype=torch.long))
        model_save = Pipeline([
            ('net', model),
        ])
        with open('models/torch_lstm.pkl','wb') as f:
            pickle.dump(model_save,f)
        
        self.models['LSTM'] = model
    
    def train_rcnn(self,class_instance,params):
        #same as conv1d
        print("Start training LSTM-CNN")
        model = NeuralNetClassifier(module = class_instance,
                                    module__out_channels = params['out_channels'],
                                    module__hidden_size = params['hidden_size'],
                                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                    criterion = nn.CrossEntropyLoss,
                                    optimizer = torch.optim.Adam,
                                    lr = 0.002,
                                    callbacks=[('tr_acc', callbacks.EpochScoring('accuracy',
                                                                                lower_is_better=False,
                                                                                on_train=True,
                                                                                name='train_acc'))],
                                    batch_size = 1000,
                                    max_epochs= 90)
        Xt,yt = self.get_tensors()
        Xtt = Xt.reshape(-1, 1, params['shape'][0], params['shape'][1])
        model.fit(Xtt.to(torch.float),yt.to(dtype=torch.long))
        model_save = Pipeline([
            ('net', model),
        ])
        with open('models/torch_rcnn.pkl','wb') as f:
            pickle.dump(model_save,f)
        
        self.models['RCNN'] = model




    
            

