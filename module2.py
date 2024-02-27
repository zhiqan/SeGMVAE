# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 09:13:16 2022

@author: Owner
"""


import torch
import torch.nn as nn


'''
class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.last_hidden_size = 2*2*hidden_size
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc31 = nn.Linear(64, hidden_size)

    def forward(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        return self.fc31(h3)

class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size   
        self.fc4 = nn.Linear(hidden_size, 64)
        self.fc5 = nn.Linear(64, 256)
        self.fc6 = nn.Linear(256, 512)
        self.fc7 = nn.Linear(512, 1024)
    def forward(self, z):
        h4 = F.leaky_relu(self.fc4(z))
        h5 = F.leaky_relu(self.fc5(h4))
        h6 = F.leaky_relu(self.fc6(h5))
        return torch.sigmoid(self.fc7(h6))

'''


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.last_hidden_size = 2* 2 * hidden_size

        self.encoder = nn.Sequential(
           nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # -1, hidden_size, 2, 2
        h = self.encoder(x)
        # -1, hidden_size*2*2
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )

    def forward(self, h):
        # -1, hidden_size, 2, 2
        #h = h.view(-1, self.hidden_size, 2,2)
        # -1, 1, 28, 28
        x = self.decoder(h)
        return x