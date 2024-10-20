"""
Model modules definition.
Modified from https://github.com/ayushtues/ADDA_pytorch
"""
import torch
import torch.nn as nn

class LeNet_Encoder(nn.Module):
    def __init__(self):
        super(LeNet_Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.flatten(out)
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

class LeNet_Decoder(nn.Module):
    """
    This is the decoder module used by the target dataset expert model. 
    Together with the target encoder they would form an autoencoder.
    """
    def __init__(self):
        super(LeNet_Decoder, self).__init__()
        self.unflatten = nn.Unflatten(1, (16, 4, 4))
        
        self.relu = nn.LeakyReLU()
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_trans1 = nn.ConvTranspose2d(16, 6, 5)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_trans2 = nn.ConvTranspose2d(6, 1, 5)
        
    def forward(self, input):
        out = self.unflatten(input)
        out = self.up1(out)
        out = self.relu(out)
        out = self.conv_trans1(out)
        out = self.up2(out)
        out = self.relu(out)
        out = self.conv_trans2(out)
        return out

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
