import torch as torch
import torch.nn as nn
from skorch import NeuralNet, callbacks, NeuralNetClassifier
from sklearn.pipeline import Pipeline
import pickle

class Train():
    def __init__(self,
                 network,
                 criterion = torch.nn.CrossEntropyLoss,
                 optimizer = torch.optim.SGD,
                 lr = 0.01,
                 device = torch.device("cpu")):
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.network = network

    def skorch_train(self, X, y, max_epochs=5):
        nnc = NeuralNetClassifier(
            module=self.network,
            device=self.device,
            criterion=self.criterion,
            optimizer=self.optimizer,
            optimizer__momentum=0.9,
            lr=self.lr,
            callbacks=[
                ('tr_acc', callbacks.EpochScoring(
                    'accuracy',
                    lower_is_better=False,
                    on_train=True,
                    name='train_acc',
                )),
            ],
            max_epochs= max_epochs
        )

        nnc.fit(X,y)
        self.nnc = nnc
        return nnc

    def save_model(self, path):
        model = Pipeline([
            ('net', self.nnc),
        ])
        if path.split(".")[1]!='pkl':
            raise Exception("Invalid file format")
        with open(path, 'wb') as f:
            pickle.dump(model, f)






