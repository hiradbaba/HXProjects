#main application to predict the class of a given image
from preprocess import Dataset
from evaluation import Evaluation
from training import Train
from model import Network,CBR_Network
from torch.utils.data import DataLoader
import torch

ds = Dataset(".\\csv\\train.csv",(32,32))
dt = Dataset(".\\csv\\test.csv",(32,32))


train_dl = DataLoader(ds, 16200, shuffle=True)
test_dl = DataLoader(dt, 1500, shuffle=False)
print("Datasets are loaded to the application!")

X_test, y_test = next(iter(test_dl))
print("Test datasets are assigned")

network = CBR_Network()
print("Model is created!")


print("Please wait while the dataset is loading...")
trainer = Train(network,device=torch.device('cuda'))
X, y = next(iter(train_dl))
net = trainer.skorch_train(X=X, y=y, max_epochs=15)
print("Model is trained!")

trainer.save_model('models/cbr_skorch.pkl')
print("Model is saved!")
y_pred = net.predict(X_test)
evalutaion = Evaluation(X_test=X_test, y_pred=y_pred, y_test=y_test, net = net)
evalutaion.evaluate()
evalutaion.history_stats()


