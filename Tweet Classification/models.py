import torch
import torch.nn as nn

class Conv1dText(nn.Module):
    def __init__(self):
        super(Conv1dText,self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Conv1d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3)
        )
        self.clf = nn.Sequential(
            nn.Linear(320,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,16),
            nn.ReLU(inplace=True),
            nn.Linear(16,3),
           nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.feature_layer(x)
        x = x.view(x.size(0),-1)
        x = self.clf(x)
        return x

class CBR_Network(nn.Module):
    def __init__(self):
        super(CBR_Network,self).__init__()
        self.feature_layer = nn.Sequential(
        nn.Conv2d(1,32,kernel_size=3,stride=1,padding=(1,1)),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2,stride=2),
        nn.Conv2d(32,64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        nn.Conv2d(64,64,kernel_size=3,padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2,2)
        )

        self.clf = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512,10),
            nn.Linear(10,3),
            nn.Softmax(dim=1)
            )

    def forward(self,x):
        x = self.feature_layer(x)
        x = x.view(x.size(0),-1)
        x = self.clf(x)
        
        return x

class RecurrentNN(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, seq_length, num_classes):
        super(RecurrentNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size
                          , num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(seq_length*hidden_size*2, num_classes)

    def forward(self, X):
        out, hidden = self.rnn(X, None)
        output = self.fc(out.reshape(out.shape[0], -1))
        return output


class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, seq_length, num_classes):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(seq_length*hidden_size*2, num_classes)

    def forward(self, X):
        out, (hidden, _) = self.lstm(X, None)
        output = self.fc(out.reshape(out.shape[0], -1))
        return output

class ConvRNN(nn.Module):
    
    def __init__(self, out_channels, hidden_size):
        super(ConvRNN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=out_channels,
                              kernel_size=3, padding=1, padding_mode='zeros'),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(kernel_size=2,stride=2))
        self.fc1 = nn.Linear(out_channels, 1)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size,
                          num_layers=2, batch_first=True, bidirectional=True)
        seq_length = 10
        num_classes = 3
        self.fc2 = nn.Linear(seq_length*hidden_size, num_classes)
        self.out_channels = out_channels
    def forward(self, X):
        out = self.conv(X)
        out = self.fc1(out.reshape(out.shape[0], -1, self.out_channels))
        outr, (hidden,_) = self.lstm(out.squeeze().reshape(out.shape[0], 5, 2), None)
        output = self.fc2(outr.reshape(outr.shape[0], -1))
        return output