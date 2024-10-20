"""
Model modules definition in addition to the gradient reversal layer.
Modified from https://github.com/ayushtues/ADDA_pytorch
"""

import torch
import torch.nn as nn
from torch.autograd import Function

class ReverseLayerF(Function):
    """
    Gradient reversal layer.
    Copied from https://github.com/fungtion/DANN
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class LeNet_Encoder(nn.Module):
    def __init__(self):
        super(LeNet_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, input, alpha):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.flatten(out)
        reverse_feature = ReverseLayerF.apply(out, alpha)
        return out, reverse_feature
    
class Discrminator(nn.Module):
    def __init__(self):
        super(Discrminator, self).__init__()
        self.fc1 = nn.Linear(256,180)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(180,126)
        self.fc3 = nn.Linear(126,88)
        self.fc4 = nn.Linear(88,2)
        self.relu = nn.LeakyReLU()

    def forward(self ,input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        return out
    
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, input):
        out1 = self.relu(self.fc1(input))
        out1 = self.dropout(out1)
        out2 = self.relu(self.fc2(out1))
        out2 = self.dropout(out2)
        out3 = self.relu(self.fc3(out2))
        soft_out = out3
        return out2, soft_out

class Class_Projector(torch.nn.Module):
    """
    This is the module used to apply the proposed class-cluster projection design pattern.
    """
    def __init__(self):
        super(Class_Projector, self).__init__()
        self.fc1 = nn.Linear(84,58)
        self.fc2 = nn.Linear(58,10)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU()

    def forward(self,input):
        out = self.relu(self.fc1(input))
        out = self.relu(self.fc2(out))
        out = self.softmax(out)
        return out
