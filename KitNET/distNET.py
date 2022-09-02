import numpy as np
# import KitNET.dA as AE
import KitNET.autoencoder as AE
import KitNET.corClust as CC
import pdb, traceback

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document

class distNET:
    #n: the number of features in your input dataset (i.e., x \in R^n)
    #m: the maximum size of any autoencoder in the ensemble layer
    #AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    #FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    #learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    #hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    #feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(self,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, corruption_level=0.0, feature_map = None):
        # Parameters:
        self.AD_grace_period = AD_grace_period
        if FM_grace_period is None:
            self.FM_grace_period = AD_grace_period
        else:
            self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.corruption_level = corruption_level
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n
        self.msg = ''

        # Variables
        self.n_trained = 0 # the number of training instances so far
        self.n_executed = 0 # the number of executed instances so far
        self.v = feature_map
        self.clusters = []
        if self.v is None:
            print("\nFeature-Mapper: train-mode, Anomaly-Detector: off-mode")
            pass
        else:
            self.__createAD__()
            # print("\nFeature-Mapper: execute-mode, Anomaly-Detector: train-mode")
        self.FM = CC.corClust(self.n) #incremental feature cluatering for the feature mapping process
        # self.ensembleLayer = []
        self.outputLayer = None
        self.models = []

    #If FM_grace_period+AM_grace_period has passed, then this function executes KitNET on x. Otherwise, this function learns from x.
    #x: a numpy array of length n
    #Note: KitNET automatically performs 0-1 normalization on all attributes.
    def process(self,x):
        self.msg = ''
        if self.n_trained <= self.FM_grace_period:
            return  self.build_clusters(x) , self.msg #If the FM are in train-mode

        elif self.n_trained <= self.FM_grace_period + self.AD_grace_period: #If the AD are in train-mode
            score = self.train(x)
            # print(self.m)
            # return 0.0, self.msg
            return score , self.msg

        else:                                   #If both the FM and AD are in execute-mode
            return self.execute(x), self.msg 
            

    #force train KitNET on x
    #returns the anomaly score of x during training (do not use for alerting)

    def build_clusters(self,x):
        score = 0
        # if self.n_trained <= self.FM_grace_period and self.v is None: #If the FM is in train-mode, and the user has not supplied a feature mapping
        #update the incremetnal correlation matrix
        self.FM.update(x)
        if self.n_trained <self.FM_grace_period:
            if self.n_trained%10 ==0:
                # print(self.n_trained, self.FM.cluster(self.m))
                self.clusters.append(self.FM.cluster(self.m))


        # pdb.set_trace()
        import copy

        if self.n_trained == self.FM_grace_period: #If the feature mapping should be instantiated
            self.v = self.FM.cluster(self.m)
            self.clusters.append(self.v)
            # for i in range(6):
            #     self.clusters.append(copy.deepcopy(self.v))
            # print(self.v)
            # import sys

            # sys.exit()

            

            for i in range(len(self.clusters)):
                self.models.append(self.__createAD__(self.clusters[i]) )

            # print("The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders.")
            # print("Feature-Mapper: execute-mode, Anomaly-Detector: train-mode")
            self.msg = "The Feature-Mapper found a mapping: "+str(self.n)+" features to "+str(len(self.v))+" autoencoders."
            self.msg +="\nFeature-Mapper: execute-mode, Anomaly-Detector: train-mode"
            # pdb.set_trace()
        self.n_trained += 1

        return 0


    def train(self, x):
        score =[]
        if self.n_trained <= self.FM_grace_period and self.v is None: #If the FM is in train-mode, and the user has not supplied a feature mapping
            raise RuntimeError('Feature mapping not available. Try running process(x) instead.')
            # pdb.set_trace()
        else: #train
            ## Ensemble Layer
            for i, cluster in enumerate(self.clusters):
                score.append(self.train_one(x, self.models[i], cluster) )
            # score = score/i
            self.n_trained += 1
        return score


    def train_one(self,x, model, cluster):        
        ensembleLayer   = model[0]
        outputLayer     = model[1]
        S_l1 = np.zeros(len(ensembleLayer))
        for a in range(len(ensembleLayer)):
            # make sub instance for autoencoder 'a'
            xi = x[cluster[a]]
            try:
                S_l1[a] = ensembleLayer[a].train(xi)
            except Exception as e:
                traceback.print_exc()
                pdb.set_trace()
            
        ## OutputLayer
        score = outputLayer.train(S_l1)
        if self.n_trained == self.AD_grace_period+self.FM_grace_period:
            self.msg = "Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode"
            # print("Feature-Mapper: execute-mode, Anomaly-Detector: execute-mode")
        return score

    #force execute KitNET on x
    def execute(self, x):
        score =[]
        if self.v is None:
            raise RuntimeError('KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead.')
        else: #train
            ## Ensemble Layer
            for i, cluster in enumerate(self.clusters):
                score.append(self.execute_one(x, self.models[i], cluster))      
            # score = score/i
            self.n_executed += 1
        return score


    def execute_one(self,x, model, cluster):        
        ensembleLayer   = model[0]
        outputLayer     = model[1]
        S_l1 = np.zeros(len(ensembleLayer))
        for a in range(len(ensembleLayer)):
            # make sub inst
            xi = x[cluster[a]]
            S_l1[a] = ensembleLayer[a].execute(xi)
        ## OutputLayer
        # pdb.set_trace()
        return outputLayer.execute(S_l1)

    def __createAD__(self, clusters):
        # pdb.set_trace()
        # construct ensemble layer
        ensembleLayer = []#self.ensembleLayer
        # print(f'{clusters}, {len(clusters)=}')
        for map in clusters:
            params = AE.dA_params(n_visible=len(map), n_hidden=0, lr=self.lr, corruption_level=self.corruption_level, gracePeriod=0, hiddenRatio=self.hr)
            ensembleLayer.append(AE.dA(params))

        # construct output layer
        params = AE.dA_params(len(clusters), n_hidden=0, lr=self.lr, corruption_level=0, gracePeriod=0, hiddenRatio=self.hr)
        outputLayer = AE.dA(params)

        return [ensembleLayer, outputLayer]

# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.