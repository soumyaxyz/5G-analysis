from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import math, re
import copy
from sklearn import utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable 
from sklearn.decomposition import PCA
import pdb, traceback

def append_to(var, value):
    if var == None:
      return value
    else:
      return torch.cat((var,value),0)

def pca(data_train, data_test, pca = None, n = 10):
  # pdb.set_trace()
  if pca is None:  
    pca = PCA(n_components=n)
    data_train = pca.fit_transform(data_train)
    data_test = pca.transform(data_test)
    return data_train, data_test, pca
  else:
    data_test = pca.transform(data_test)
    return data_test


class FiveG_data_loader():
  """docstring for Task1_data_loader"""
  def __init__(self, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, 
        shuffle = False, noise_from_class = 1, mask= [0],# [,*range(7,32), 0, 1, 2, 3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 26, 29, 30, 31]
        add_synthetic=True, guard=0 ):   #,7,8,18,17,*range(20,32)
    self.poison_persent   = poison_persent
    self.training_fraction  = training_fraction
    assert placement_percent >=0 and placement_percent <=1, 'placement_percent should be between 0(leftmost) and 1(rightmost)'
    self.placement_percent  = placement_percent
    self.shuffle      = shuffle
    data_subfolder = '5G/training_data'
    self.data_files = [ #data_subfolder+"/SNR_data_10K__10feet_0degree.csv",    # normal 
              data_subfolder+"/SNR_data_5000_10feet_0degree.csv",   # normal 
                data_subfolder+"/SNR_data_5000_10feet_45degree.csv",
                data_subfolder+"/SNR_data_5000_10feet_90degree.csv",
                data_subfolder+"/SNR_data_5000_10feet_180degree.csv",
              data_subfolder+"/SNR_data_5000_20feet_0degree.csv",     
                data_subfolder+"/SNR_data_5000_20feet_45degree.csv",
                data_subfolder+"/SNR_data_5000_20feet_90degree.csv",              
                data_subfolder+"/SNR_data_5000_20feet_180degree.csv",
              data_subfolder+"/SNR_data_5000_30feet_0degree.csv",     
                data_subfolder+"/SNR_data_5000_30feet_45degree.csv",
                data_subfolder+"/SNR_data_5000_30feet_90degree.csv",              
                data_subfolder+"/SNR_data_5000_30feet_180degree.csv",
              data_subfolder+"/SNR_data_5000_40feet_0degree.csv",     
                data_subfolder+"/SNR_data_5000_40feet_45degree.csv",
                data_subfolder+"/SNR_data_5000_40feet_90degree.csv",              
                data_subfolder+"/SNR_data_5000_40feet_180degree.csv",
              data_subfolder+"/SNR_data_5000_50feet_0degree.csv",     
                data_subfolder+"/SNR_data_5000_50feet_45degree.csv",
                data_subfolder+"/SNR_data_5000_50feet_90degree.csv",              
                data_subfolder+"/SNR_data_5000_50feet_180degree.csv"              
          ]
    test_subfolder = '5G/testing_data'      
    self.test_files = [ #data_subfolder+"/SNR_data_10K__10feet_0degree.csv",    # normal 
              test_subfolder+"/SNR_data_1000_25feet_0degree.csv",     
                test_subfolder+"/SNR_data_1000_25feet_45degree.csv",
                test_subfolder+"/SNR_data_1000_25feet_90degree.csv",              
                test_subfolder+"/SNR_data_1000_25feet_180degree.csv",
              test_subfolder+"/SNR_data_1000_35feet_0degree.csv",     
                test_subfolder+"/SNR_data_1000_35feet_45degree.csv",
                test_subfolder+"/SNR_data_1000_35feet_90degree.csv",              
                test_subfolder+"/SNR_data_1000_35feet_180degree.csv"              
          ]
    # self.data_files = self.data_files[:3] # masking out all but the first 4 classes
    self.filename= []
    for i in range(len(self.data_files)):
      self.filename.append(self.data_files[i][18:-4])
    assert noise_from_class >=0 , 'class 0 must be clean data, cannot be noise'
    assert noise_from_class <len(self.data_files) , 'class 0 must be clean data, cannot be noise'
    self.first = noise_from_class-1
    self.mask = mask
    # pdb.set_trace()
    self.add_synthetic = add_synthetic
    self.guard = guard
    self.feature_names = ['sl_no','Rx Power', 'Rx Power Average', 'Rx Power OTA', 'Rx Power Average OTA', 'Rx Signal to Noise Ratio', 
                'Rx Average SNR', 'Rx SNR over Gi64', 'Rx Average SNR over Gi64 ', 'Rx Packet Error Rate (PER)', 
                'Tx Missed Ack Rate', 'Rx AGC attenuation', 'Rx average AGC attenuation', 'Tx MCS', 'Rx MCS', 
                'Local Device Tx sector', 'Local Device Rx sector', 'TXSS periods SSW frame recv', 
                'RXSS periods SSW frame recv', 'Best TXSS sector ', 'Best RXSS sector ', 
                'Ethernet Packets Sent', 'Bytes Sent ', 'Bytes Transmitted ', 'Bytes Transmitted MAC Frm', 
                'Announce frames sent', 'BI announce acked', 'BI announce not acked', 'MPDUs received ', 
                'MPDUs transmitted ', 'Inactive time', 'BH2 temperature'
              ]
    self.feature_names = np.delete(self.feature_names,mask)
    
    
  def loadfile(self, filename, mask = []):
    df  = pd.read_csv(filename, header = None, skiprows = self.guard)
    data = df.values
    # pdb.set_trace()
    data = np.delete(data,mask,1)
    tokens =  re.findall(r'\d+', filename)
    try:
      label = int(tokens[3])
    except IndexError as e:
      label = 0
    label = [int(tokens[2])*-1, label]  # denoting the distance as negative and degrees as posative
    # print(label)
    return data, label

  def loadSupervisedData(self, test_only= False):
    if test_only:
      data_files = self.test_files
    else:
      data_files = self.data_files
    data_count = len(data_files)

    training_size_equalized = np.inf
    total_training_size = 0

    
    if test_only:
      training_size = 0
    else:
      for i in range(data_count):
        class_data, _ = self.loadfile(data_files[i], self.mask)
        training_size   = round(class_data.shape[0]*self.training_fraction)
        total_training_size+= training_size
        if training_size_equalized> training_size:
          training_size_equalized = training_size


      # holdout = round(total_training_size/10)
      training_size = training_size_equalized

    unique_class = set()


    for i in range(data_count):
      # if i == 0:
      #   training_size = holdout+training_size_equalized
      # else:
      #   training_size = training_size_equalized
      class_data, class_label = self.loadfile(data_files[i], self.mask)
      for l in class_label:
        unique_class.add(l)
      # training_size   = round(class_data.shape[0]*self.training_fraction)
      # print(f'{training_size=},{class_data.shape[0]=}')
      if i == 0 :
        train_data      = class_data[:training_size]
        train_labels    = [class_label]*(training_size)

        test_data       = class_data[training_size:]
        test_labels     = [class_label]*(class_data.shape[0]-training_size)  
                
      else:
        # pdb.set_trace()
        train_data      = np.vstack((train_data, class_data[:training_size]))       
        train_labels    = np.vstack((train_labels, [class_label]*(training_size)))

        test_data       = np.vstack((test_data, class_data[training_size:]))        
        test_labels     = np.vstack((test_labels, [class_label]*(class_data.shape[0]-training_size)))

    # pdb.set_trace()
    if test_only:
      self.unique_test_class = sorted(unique_class)
      self.test_key = {idx:key for key,idx in enumerate( self.unique_test_class)}
      self.training_size = train_labels.shape[0]
      self.training_size_initial = self.training_size
    else:
      self.unique_class = sorted(unique_class)
      self.class_key = {idx:key for key,idx in enumerate( self.unique_class)}
    
    
    if self.poison_persent>0:     
      # data, labels  = self.poison(data, labels)
      raise NotImplementedError

    
    if self.shuffle:
      # data, labels  = self.shuffle_training_data_only(data, labels)
      raise NotImplementedError

    if test_only:
      test_data = self.scaler.transform(test_data)
    else:
      self.scaler = MinMaxScaler()      
      self.scaler.fit(train_data)

      train_data = self.scaler.transform(train_data)
      test_data = self.scaler.transform(test_data)

    if self.add_synthetic:
      # normalized_data, labels = self.add_synthetic_training_data(normalized_data, labels)
      raise NotImplementedError




    train_data    = train_data.astype(np.float64) 
    test_data     = test_data.astype(np.float64)  

    # if test_only:
    #   print('in dataset')
    #   pdb.set_trace()


    if test_only:
      test_data = pca(train_data, test_data, self.pc)
      return test_data, test_labels
    else:
      train_data, test_data, self.pc = pca(train_data, test_data)
      return train_data, test_data, train_labels, test_labels


class FiveG_Dataset(Dataset):
  
  def __init__(self, mode, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, shuffle = False, noise_from_class = 1,  add_synthetic = False, guard = 0):
    self.mode               = mode
    self.poison_persent     = poison_persent
    self.training_fraction  = training_fraction
    self.placement_percent  = placement_percent
    self.shuffle      = shuffle
    assert noise_from_class >=0 , 'class 0 must be clean data, cannot be noise'
    self.adversarial_class  = noise_from_class
    self.add_synthetic      = add_synthetic
    self.guard              = guard

    # self.get_val_data = False
    # self.get_test_data = False

    # pdb.set_trace()

    data_loader =   FiveG_data_loader(self.poison_persent, self.training_fraction, self.placement_percent, self.shuffle, self.adversarial_class, add_synthetic = self.add_synthetic, guard = self.guard)
    
    
    train_data, self.test_data, train_labels, self.test_labels      = data_loader.loadSupervisedData()
    self.train_data, self.val_data, self.train_labels,  self.val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)
    self.test_only_data, self.test_only_labels = data_loader.loadSupervisedData(test_only= True)
    
    class_list = [key for key in data_loader.class_key]
    partion_point = class_list.index(0)
    self.class_key_d = { i : class_list[i]*-1 for i in range(partion_point ) }
    self.class_key_a = { i-partion_point : class_list[i] for i in range(partion_point,len(class_list)) }
    self.class_map_d = {v: k for k, v in self.class_key_d.items()}
    self.class_map_a = {v: k for k, v in self.class_key_a.items()}
    self.min_d = min(self.class_key_d.values()) 
    self.max_d = max(self.class_key_d.values())
    self.min_a = min(self.class_key_a.values()) 
    self.max_a = max(self.class_key_a.values())
    # if self.mode == 'test_only':
    #   pdb.set_trace()
    # print(self.max_d, self.min_d, self.max_a, self.min_a)
    

  def normalize_dist(self, x):
    return (x - self.min_d)/(self.max_d- self.min_d)*10

  def normalize_angle(self, x):
    return (x - self.min_a)/(self.max_a- self.min_a)*10
 

  def __len__(self):
    if self.mode == 'test':
      return len(self.test_labels)
    elif self.mode == 'test_only':
      return len(self.test_only_labels)
    elif self.mode == 'val':
      return len(self.val_labels)
    else:
      return len(self.train_labels)

  def __getitem__(self,idx):
    if self.mode == 'test':
      data =  self.test_data[idx]
      label = self.test_labels  
    elif self.mode == 'test_only':
      data =  self.test_only_data[idx]
      label = self.test_only_labels
    elif self.mode =='val':
      data =  self.val_data[idx]
      label = self.val_labels
    else:
      data =  self.train_data[idx]
      label = self.train_labels

    label1 = label[idx][0]*-1
    label2 = label[idx][1]
    label_ =label[idx] 
    # print(label1, self.class_map_d)

    size_d = len(self.class_key_d)
    size_a = len(self.class_key_a)
    try:
      mapd = self.class_map_d[label1]
      label_d = np.eye(size_d, dtype='uint8')[mapd]
    except KeyError as e:
      mapd = -1
      label_d = np.zeros(size_d, dtype='uint8')
    try:
      mapa = self.class_map_d[label2]
      label_a = np.eye(size_a, dtype='uint8')[mapa]
    except KeyError as e:
      mapa = -1
      label_a = np.zeros(size_a, dtype='uint8')
    

    # onehot_label = np.sum([label_1,label_2], axis=0)
    # print(self.class_map, label2)
    # pdb.set_trace()
    #'labels_dist': self.class_map_d[label1], 'labels_angle': self.class_map_a[label2]}

    sample = {'data':data, 'labels_d': label_d.astype('f'), 'labels_a': label_a.astype('f'),
    'labels_dist': mapd, 'labels_angle': self.class_map_a[label2], 
    'labels_Dist': self.normalize_dist(label1), 'labels_Angle': self.normalize_angle(label2),
    'raw_dist': label1, 'raw_angle': label2}  
    return sample   


class LSTMnetwork(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length=128):
      super(LSTMnetwork, self).__init__()
      self.num_classes = num_classes #number of classes
      self.num_layers = num_layers #number of layers
      self.input_size = input_size #input size
      self.hidden_size = hidden_size #hidden state
      self.seq_length = seq_length #sequence length

      self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=True) #lstm
      self.fc_1 =  nn.Linear(hidden_size, seq_length) #fully connected 1
      self.relu = nn.ReLU()

      self.fc_a = nn.Linear(seq_length, num_classes[0]) #fully connected last layer
      

      self.fc_b = nn.Linear(seq_length, num_classes[1]) #fully connected last layer
    
    def forward(self,x):
      x = torch.reshape(x.float(),   (x.shape[0], 1, x.shape[1]))
      h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
      c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
      # Propagate input through LSTM
      output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
      hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
      out = self.relu(hn)
      out = self.fc_1(out) #first Dense

      out_a = self.relu(out) #relu
      out_a = self.fc_a(out_a) #Final Output_a

      out_b = self.relu(out) #relu
      out_b = self.fc_b(out_b) #Final Output_b
      return out_a, out_b


class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
      super(LinearRegression, self).__init__()
      self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
      out = self.linear(x)
      return out

class Experiment():
  def __init__(self):
    self.dataset=None
    

  def load_data(self):
    self.dataset = FiveG_Dataset('train')
    # train_set = self.dataset
    # self.dataset.set_val_data_mode()
    val_set = FiveG_Dataset('val')
    # self.dataset.set_test_data_mode()
    test_set = FiveG_Dataset('test')

    test_only_set = FiveG_Dataset('test_only')

    self.train_loader  = DataLoader(self.dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    self.val_loader    = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    self.test_loader   = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)    
    self.test_only_loader   = DataLoader(test_only_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    self.num_classes = [len(self.dataset.class_key_d)] #number of output distance classes 
    self.num_classes.append( len(self.dataset.class_key_a) )#number of output distance classes 
    self.class_map_d = self.dataset.class_map_d
    self.class_map_a = self.dataset.class_map_a
    # pdb.set_trace()

  def train_model(self, model,  optimizer,  num_epochs=100, patience = 5):
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    # criterion = lambda y_hat,y: F.cross_entropy(y_hat,y)
    min_val_loss= np.inf
    stagnant = 0
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
      model.train()
      sum_loss = 0.0
      total = 0
      for sample in tqdm (self.train_loader, leave=False):
        # pdb.set_trace()
        x = sample['data']
        y_d = sample['labels_d']
        y_a = sample['labels_a']

        y_d_hat, y_a_hat = model.forward(x) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        # pdb.set_trace()
        # obtain the loss function
        loss_a = criterion(y_a_hat, y_a)
        loss_d = criterion(y_d_hat, y_d)
        loss = torch.add(loss_a, loss_d)

        loss.backward() #calculates the loss of the loss function     
        optimizer.step() #improve from loss, i.e backprop
        sum_loss += loss.item()*y_d.shape[0]
        total += y_d.shape[0]

      val_loss, val_acc, val_rmse = self.validation_metrics(model)
      if min_val_loss> val_loss:
        min_val_loss= val_loss
        stagnant = 0
      else:
        stagnant +=1
      if stagnant >= patience:
        break

      # if epoch % 5 == 0:
      # print(f"{min_val_loss=}, {stagnant=}, {patience=}")
      # pbar.set_description("train loss %.3f, val loss %.3f, val acc %.3f, val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
      pbar.set_postfix({'train loss %.3f': sum_loss/total, 'val loss %.3f' : val_loss})
      with open('stats_multihead_mrmr.csv', 'a') as file:
        file.write(" %.4f, %.4f, %.4f, %.4f\n" % (sum_loss/total, val_loss, val_acc, val_rmse))


  def validation_metrics(self, model):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    # criterion = lambda outputs,y: F.cross_entropy(outputs,y)
    for sample in tqdm (self.val_loader, leave=False):
      # pdb.set_trace()
      x = sample['data']
      y_d = sample['labels_d']
      y_a = sample['labels_a']
      y__d = sample['labels_dist']
      y__a = sample['labels_angle']

      y_d_hat, y_a_hat = model.forward(x) #forward pass
      # pdb.set_trace()
      # obtain the loss function
      loss_a = criterion(y_a_hat, y_a)
      loss_d = criterion(y_d_hat, y_d)
      loss = torch.add(loss_a, loss_d)

     
      
      # pdb.set_trace()
      pred_d = torch.max(y_d_hat, 1)[1]
      pred_a = torch.max(y_a_hat, 1)[1]
      correct += (pred_d == y__d).float().sum()
      correct += (pred_d == y__d).float().sum()
      total += 2*y_a.shape[0]
      sum_loss += loss.item()*y_a.shape[0]
      sum_rmse += np.sqrt(mean_squared_error(pred_d, y__d))*y_d.shape[0]
      sum_rmse += np.sqrt(mean_squared_error(pred_a, y__a))*y_a.shape[0]
    return sum_loss/(total/2), correct/total, sum_rmse/total

  def train_Regression(self, model,  regression_d, regression_a, optimizer,  num_epochs=100, patience = 5):
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    # criterion = lambda y_hat,y: F.cross_entropy(y_hat,y)
    min_val_loss= np.inf
    stagnant = 0
    pbar = tqdm(range(num_epochs))
    model.eval()
    regression_d.train()
    regression_a.train()
    for epoch in pbar:
      sum_loss = 0.0
      total = 0
      i=0
      for sample in tqdm (self.train_loader, leave=False):
        # sample = next(iter(self.train_loader))
        # pdb.set_trace()
        x = sample['data']
        y_d = sample['labels_dist']
        y_a = sample['labels_dist']
        y_D = sample['labels_Dist']
        y_A = sample['labels_Angle']
        y_D = y_D.type(torch.FloatTensor)
        y_A = y_A.type(torch.FloatTensor)

        y_d_, y_a_ = model.forward(x) #forward pass

        y_d_hat = regression_d.forward(y_d_)
        y_a_hat = regression_a.forward(y_a_)
        optimizer.zero_grad()

        y_d_hat = y_d_hat.view(y_D.shape)
        y_a_hat = y_a_hat.view(y_A.shape)


        #caluclate the gradient, manually setting to 0
        # pdb.set_trace()
        # obtain the loss function
        loss_d = criterion(y_d_hat, y_D)
        loss_a = criterion(y_a_hat, y_A)
        loss = torch.add(loss_a, loss_d)

        loss.backward() #calculates the loss of the loss function     
        optimizer.step() #improve from loss, i.e backprop
        sum_loss += loss.item()*y_D.shape[0]
        total += y_D.shape[0]

      # val_loss, val_acc, val_rmse = self.validation_metrics(model)
      val_loss = loss.item()
      if min_val_loss> val_loss:
        min_val_loss= val_loss
        stagnant = 0
      else:
        stagnant +=1
      if stagnant >= patience:
        print("\nearly stopping at epoch",epoch)
        break

      # if epoch % 5 == 0:
      # print(f"{min_val_loss=}, {stagnant=}, {patience=}")
      # pbar.set_description("train loss %.3f, val loss %.3f, val acc %.3f, val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
      pbar.set_postfix({'train loss %.3f': sum_loss/total, 'val loss %.3f' : val_loss})
      with open('regression_stats_multihead_pca.csv', 'a') as file:
        file.write(" %.4f, %.4f\n" % (sum_loss/total, val_loss))
    print(torch.round(y_d_hat),'\n', y_d)
    print(torch.round(y_a_hat),'\n', y_a)
    self.save_model(regression_d, optimizer, 'saved_regression_dist_model_pca.pkl')
    self.save_model(regression_a, optimizer, 'saved_regression_angle_model_pca.pkl')
    pdb.set_trace()

  def test_predict(self, model, data_loader = None):
    if data_loader is None:
      data_loader = self.test_loader
    model.eval()
    pred_A = None
    pred_D = None 
    Y_a = None 
    Y_d = None
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    # criterion = lambda outputs,y: F.cross_entropy(outputs,y)
    for sample in tqdm (data_loader, leave=False):
      # sample = next(iter(self.test_only_loader))
      # pdb.set_trace()
      x = sample['data']
      y_d = sample['labels_d']
      y_a = sample['labels_a']
      y__d = sample['labels_dist']
      y__a = sample['labels_angle']

      y_d_hat, y_a_hat = model.forward(x) #forward pass
      # pdb.set_trace()
      # obtain the loss function
      loss_a = criterion(y_a_hat, y_a)
      loss_d = criterion(y_d_hat, y_d)
      loss = torch.add(loss_a, loss_d)

      
      pred_d = torch.max(y_d_hat, 1)[1]
      pred_a = torch.max(y_a_hat, 1)[1] 
      pred_D = append_to(pred_D, pred_d)
      pred_A = append_to(pred_A, pred_a)
      Y_d = append_to(Y_d, y__d)
      Y_a = append_to(Y_a, y__a)
    return pred_A, pred_D, Y_a, Y_d

  def save_model(self, model, optimizer, filename='saved_multihead_model_MRMR.pkl'):
    torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),}, 
      filename)

  def load_model(self, model, optimizer, filename='saved_multihead_model_pca.pkl'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

  def show_results(self, pred_A, pred_D, Y_a, Y_d):
    distance_acc = accuracy_score(Y_d, pred_D)
    angle_acc = accuracy_score(Y_a, pred_A)

    print(f'distance accuracy score :{distance_acc:.3f}')
    print(f'angle accuracy score :{angle_acc:.3f}')

    
    label_d = [key for key in self.dataset.class_key_d]
    label_rename_d = [key for key in self.class_map_d]

    label_a = [key for key in self.dataset.class_key_a]
    label_rename_a = [key for key in self.class_map_a]

    # pdb.set_trace()

    cm = confusion_matrix(Y_d, pred_D, labels=label_d)
    plot_CM(cm, label_rename_d)

    cm = confusion_matrix(Y_a, pred_A, labels=label_a)     
    plot_CM(cm, label_rename_a)


  def run(self, num_epochs = 100 ,learning_rate = 0.001, hidden_size = 2 ,num_lstm_layers = 1):

    
    if self.dataset == None:
      print('call load_data() before run()!!')
      return
      
    input_size = self.dataset.train_data.shape[1]

    model = LSTMnetwork(self.num_classes, input_size, hidden_size, num_lstm_layers) #our lstm class
    # regression_d = LinearRegression(self.num_classes[0],1) 
    # regression_a = LinearRegression(self.num_classes[1],1)

    model = model.float() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer_2 = torch.optim.SGD(model.parameters(), lr=0.01)
    # pdb.set_trace()
    model, optimizer = self.load_model(model, optimizer)

    # self.train_model(model, optimizer, num_epochs)
    # regression_d, optimizer_2 = self.load_model(regression_d, optimizer_2, 'saved_regression_dist_model_pca.pkl')
    # regression_a, _           = self.load_model(regression_a, optimizer_2, 'saved_regression_angle_model_pca.pkl')

    # self.train_Regression(model,  regression_d, regression_a, optimizer_2, num_epochs= 100,patience =20)

    # self.save_model(model, optimizer)
    # self.save_model(regression_d, optimizer_2, 'saved_regression_dist_model_pca.pkl')
    # self.save_model(regression_a, optimizer_2, 'saved_regression_angle_model_pca.pkl')

    pred_A, pred_D, Y_a, Y_d = self.test_predict(model)

    self.show_results(pred_A, pred_D, Y_a, Y_d)
    print('in run!')
    pdb.set_trace()

def plot_CM(cm, labels):
  import seaborn as sns
  import matplotlib.pyplot as plt     

  ax= plt.subplot()
  sns.heatmap(cm, annot=True, fmt='g', ax=ax)#, cmap="Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
  ax.set_xlabel('Predicted labels')
  ax.set_ylabel('True labels')
  ax.set_title('Confusion Matrix') 
  ax.xaxis.set_ticklabels(labels)
  ax.yaxis.set_ticklabels(labels)
  plt.show()

def main():
  print('\x1bc')

  experiment = Experiment()
  experiment.load_data()
  experiment.run()
  

  

if __name__ == '__main__':
  main()