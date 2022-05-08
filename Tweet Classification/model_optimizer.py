import numpy as np
import sklearn
import sklearn.model_selection
import sklearn.metrics
'''Applies hyper parameter search for [model] given data [X] and targets [y]'''
class ModelSelector:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        # storing the best estimator
        self.best_estimator = None

    '''applies grid search if random_search=False else random search. Uses 5-fold cross validation. By default uses all cores to process the search'''
    def parameter_search(self,params,random_search = True,cv=5, verbose=True,n_jobs=-1):
        print("Hyper parameter search started")
        search_cv = sklearn.model_selection.RandomizedSearchCV(self.model,params,cv=cv,verbose=verbose,n_jobs=n_jobs) if random_search else sklearn.model_selection.GridSearchCV(self.model,params,cv=cv,verbose=verbose,n_jobs=n_jobs)
        search_cv.fit(self.X,self.y)
        #stores the best result
        self.best_estimator = search_cv.best_estimator_
        return search_cv

    '''Given targets and predictions returns the classification report of the model'''
    def best_report(self,y_test,y_pred):
        if self.best_estimator is not None:
            return sklearn.metrics.classification_report(y_test,y_pred)

    '''returns the best estimator which was found in hyperparameter search'''
    def get_best_estimator(self):
        if self.best_estimator is not None:
            return self.best_estimator
        else:
            raise Exception("No model selected")