# import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
# from KitNET.utils import *
# import json

class dA_params:
    def __init__(self,n_visible = 5, n_hidden = 3, lr=0.001, corruption_level=0.0, gracePeriod = 10000, hiddenRatio=None, learning_rate=1e-4):
        self.n_visible = n_visible# num of units in visible (input) layer
        self.n_hidden = n_hidden# num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio
        self.learning_rate = learning_rate

        if self.hiddenRatio is not None:
            self.n_hidden = int(np.ceil(self.n_visible*self.hiddenRatio))

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class AE(nn.Module):
    def __init__(self, params):
        super(AE, self).__init__()
        # pdb.set_trace()
        self.encoder = nn.Linear(in_features=params.n_visible, out_features=params.n_hidden)
        self.decoder = nn.Linear(in_features=params.n_hidden, out_features=params.n_visible) 
    
    def forward(self, x):
        # x = F.relu(self.encoder(x))
        # pdb.set_trace()
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class EnsembleLayer(nn.Module):
    def __init__(self, clusters):
        super(AE, self).__init__()
        # pdb.set_trace()
        self.clusters = clusters
        self.autoencoders = []
        for i in len(self.clusters):
            self.autoencoders[i] = AE
        
    def forward(self, x):
        x_ = []
        for i, cluster in enumerate(self.clusters):
            xi = x[cluster]
            xi_ = self.autoencoders[i](xi)
            x_ = torch.cat((x_,xi_),dim=1)
        return x_



# def __createAD__(self):
#         # pdb.set_trace()
#         # construct ensemble layer
#         for map in self.v:
#             params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
#             self.ensembleLayer.append(AE.dA(params))

#         # construct output layer
#         params = AE.dA_params(len(self.v), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
#         self.outputLayer = AE.dA(params)




class dA:
    def __init__(self, params):     
        self.params = params
        # for 0-1 normlaization
        self.norm_max = np.ones((self.params.n_visible,)) * -np.Inf
        self.norm_min = np.ones((self.params.n_visible,)) * np.Inf
        self.n = 0    # epoch / packet count


        self.model = AE(self.params)
        self.trained = False


        torch.manual_seed(42)
        self.criterion = RMSELoss() # root mean square error loss
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=1e-5) 
        # print('Adam')
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.learning_rate) 

        self.model.train()

    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1
        self.rng = np.random.RandomState(1234)
        return self.rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

    def train(self, x): 
        self.n = self.n + 1
        # update norms



        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 0-1 normalize
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

        # x = x.astype(np.double)
        # x = torch.from_numpy(x)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
            tilde_x = torch.from_numpy(tilde_x)
        else:
            tilde_x = torch.from_numpy(x)

        x = torch.from_numpy(x)

        # pdb.set_trace()
        self.model.double()
        x_r = self.model(tilde_x)
        loss = self.criterion(x_r, x)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.trained = True
        return loss.item() # RMSE

    def execute(self, x): #returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            assert self.trained, 'model not trained'
            self.model.eval()
            # 0-1 normalize
            x1 = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

            self.model.double()
            x1 = torch.from_numpy(x1)

            x_r = self.model(x1)
            loss = self.criterion(x_r, x1)
            # pdb.set_trace()
            return loss.item() # RMSE

    