from torchvision import transforms, utils
from skimage import io, transform
import torch.nn as nn
from PIL import Image

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)


    def forward(self,x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CBR_Network(nn.Module):
    def __init__(self):
        super(CBR_Network,self).__init__()
        self.feature_layer = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,stride=1,padding=(1,1)),
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
            nn.Linear(64 * 8 * 8, 512),
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


class Mobilenetv1(nn.Module):
    def __init__(self, num_channels=3, num_labels=3):
        super(Mobilenetv1, self).__init__()
        self.train_sequnce = nn.Sequential(
            self.conv(3, 32, 2),
            self.conv_dw(32, 64, 1),
            self.conv_dw(64, 128, 2),
            self.conv_dw(128, 128, 1),
            self.conv_dw(128, 256, 2),
            self.conv_dw(256, 256, 1),
            self.conv_dw(256, 512, 2),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 512, 1),
            self.conv_dw(512, 1024, 2),
            self.conv_dw(1024,1024, 1),
            nn.AvgPool2d(7)
        )
        self.fc = nn.Linear(1024,num_labels)
    def conv(self,num_input,num_output,stride):
        return nn.Sequential(
                nn.Conv2d(num_input,num_output,3,stride,1),
                nn.BatchNorm2d(num_output),
                nn.ReLU(inplace=True)
        )

    def conv_dw(self,num_input,num_output,stride):
        return nn.Sequential(
                nn.Conv2d(num_input, num_input, 3, stride, 1, groups=num_input),
                nn.BatchNorm2d(num_input),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_input, num_output, 1, 1, 0),
                nn.BatchNorm2d(num_output),
                nn.ReLU(inplace=True),
            )


    def forward(self,x):
        x = self.train_sequnce(x)
        x = x.view(-1,1024)
        return self.fc(x)
