from preprocess import Dataset
from evaluation import Evaluation
from training_pytorch import Train
from model import *
from torch.utils.data import DataLoader
import torch as torch
import torch.nn as nn
import time
from sklearn.model_selection import KFold
import pickle
import numpy as np


ds = Dataset("./csv/train.csv",(32,32))
dt = Dataset("./csv/test.csv",(32,32))
# network = Network()
# mobile = Mobilenetv1()

kFold = KFold(n_splits=10, shuffle=True)
trainers = []
for fold, (train_ids, validation_ids) in enumerate(kFold.split(ds)):
    # fold = 1 , 0-10 for val and 10-100 for train
    # fold = 2 , 10-20 for val and 0-10 + 20-100 for train

    # train = 15000
    # test = 1500
    # print(validation_ids)

    cbr = CBR_Network()
    print(f'{fold} is started.')
    trainer = Train(cbr,network_name=f"cbr_network_cpu_kfold_{fold}_balanced",device = torch.device('cuda'))
    start_time = time.time()
    trainer.pytorch_train(ds, train_ids, validation_ids,max_epoch=8)
    trainers.append(trainer)
    print(f'Training the model of fold {fold} is done.')

with open(f"train_data/10_fold_training_data_fold_balanced.pkl", 'wb') as f:
    pickle.dump(trainers, f)
    print("Training data is saved.")

