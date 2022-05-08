import torch as torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import model
import numpy as np
from time import time
class Train():
    def __init__(self,
                 network,
                 criterion = torch.nn.CrossEntropyLoss,
                 optimizer = torch.optim.SGD,
                 lr = 0.01,
                 network_name = None,
                 device = torch.device('cpu')):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)
        self.lr = lr
        self.device = device
        self.network = network.to(device=device)
        if network_name is None:
            if isinstance(network,model.Network):
                self.network_name = "mini_network"
            elif isinstance(network,model.CBR_Network):
                self.network_name = "cbr_network"
            elif isinstance(network,model.Mobilenetv1):
                self.network_name = "mobilenet"
        else:
            self.network_name = network_name


    def save_model(self, path):
        return

    def pytorch_train(self, dataset, train_ids, validation_ids, batch_size = 128, max_epoch = 5):
        train_dataset = torch.utils.data.Subset(dataset, train_ids)
        validation_dataset = torch.utils.data.Subset(dataset, validation_ids)
        train_loader = DataLoader(train_dataset, batch_size=batch_size ,shuffle= True)
        validation_loader = DataLoader(validation_dataset, batch_size= batch_size ,shuffle= True)
        self.train_accuracy_avg = []
        self.validation_accuracy_avg = []
        self.validation_loss_avg = []
        self.train_loss_avg = []
        self.max_epoch = max_epoch

        torch.cuda.empty_cache()
        for epoch in range(max_epoch):
            local_train_accuracy_list = []
            local_validation_accuracy_list = []
            local_train_loss_list = []
            local_validation_loss_list = []
            start_time = time()
            for i, data in enumerate(train_loader):
                # print(data)
                inputs, labels = data
                inputs, labels = inputs.to(device=self.device),labels.to(device=self.device)
                X_validation, y_validation = next(iter(validation_loader))
                X_validation, y_validation = X_validation.to(device = self.device),y_validation.to(device = self.device)

                self.optimizer.zero_grad()

                outputs = self.network(inputs)
                loss = self.criterion(outputs, labels)
                local_train_loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()
                _, train_predict = torch.max(outputs.data, 1)
                train_correct = (train_predict == labels).sum().item()
                local_train_accuracy_list.append(train_correct / len(labels))
                local_train_loss_list.append(loss.item())

                validation_outputs = self.network(X_validation)
                validation_loss = self.criterion(validation_outputs,y_validation)
                _, validation_predict = torch.max(validation_outputs.data, 1)
                validation_correct = (validation_predict == y_validation).sum().item()
                local_validation_accuracy_list.append(validation_correct / len(y_validation))
                local_validation_loss_list.append(validation_loss.item())
            self.train_accuracy_avg.append(sum(local_train_accuracy_list)/len(local_train_accuracy_list))
            self.validation_accuracy_avg.append(sum(local_validation_accuracy_list)/len(local_validation_accuracy_list))
            self.validation_loss_avg.append(sum(local_validation_loss_list)/len(local_validation_loss_list))
            self.train_loss_avg.append(sum(local_train_loss_list)/len(local_train_loss_list))
            print('epoch: %d\ttrain_acc: %.2f\tvalid_acc: %.2f \tloss: %.3f\tdur: %.3f'%(epoch+1,self.train_accuracy_avg[-1]*100, self.validation_accuracy_avg[-1]*100, self.train_loss_avg[-1],time()-start_time))

        torch.save(self.network, f"./models/{self.network_name}.pt")




