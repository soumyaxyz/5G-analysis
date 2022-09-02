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
import pdb
from matplotlib import pyplot as plt, colors


from sklearn.decomposition import PCA

def append_to(var, value):
    if var == None:
      return value
    else:
      return torch.cat((var,value),0)

class FiveG_data_loader():
  """docstring for Task1_data_loader"""
  def __init__(self, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, 
        shuffle = False, noise_from_class = 1, mask=[0], add_synthetic=True, guard=0 ):  # #,7,8,18,17,*range(20,32)  3, 9, 10,  14, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,   4,5,8
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

  def shuffle_training_data_only(self, data, labels):
    print('shuffle training data...')
    shuffel_point   = self.benign_size
    train_data    = data[:shuffel_point]
    eval_data     = data[shuffel_point:]

    train_labels  = labels[:shuffel_point]
    eval_labels   = labels[shuffel_point:]

    train_data, train_labels = utils.shuffle(train_data, train_labels)

    # pdb.set_trace()
    data  = np.vstack((train_data, eval_data ))
    labels  = np.hstack((train_labels, eval_labels ))

    return data, labels

  def poison(self, data, labels):   
    ts    = self.training_size   
    bs    = self.benign_size  
    ps    = round(ts*(self.poison_persent/100)) #ps = poison size
    x     = min(round(self.training_size*self.placement_percent), ts-ps)  #self.placement_percent#
    x     = max(round(self.training_size*.1),x)
    # post_poison_training_size = ps*self.placement_percent
    # x = ts-ps-post_poison_training_size
    # x = self.placement_percent

    # pdb.set_trace()

    
    data  = np.vstack(( data[:x], data[bs:bs+ps], data[x:ts-ps], data[ts-ps:bs], data[bs+ps:]))
    labels  = np.hstack((labels[:x], labels[bs:bs+ps], labels[x:ts-ps], labels[ts-ps:bs], labels[bs+ps:]))  
    print('\npoison sample size:',ps, 'out of', ts,',starting at',x)

    # print(x,'clean ,followed by',ps, 'poison', ts,',followed by',ts-ps-x,"clean")

    # pdb.set_trace() 
    # aaa = ' '.join([str(int(elem)) for elem in labels])

    # print(':',ts-ps, '\t', ts,ts+ps , '\t',  ts-ps,ts,'\t', ts+ps,':')


    # bbb = aaa[:ts-ps]+ aaa[ts:ts+ps]+ aaa[ts-ps:ts]+ aaa[ts+ps:]

    return data, labels

  def add_synthetic_training_data(self, data, labels):
    gen_df = pd.read_csv('task_1/GEN.csv')
    gen =  gen_df.values
    # pdb.set_trace()
    # gen =  gen[:self.training_size]
    synthetic_size = gen.shape[0]

    x = self.training_size
    self.training_size_initial = x
    self.training_size += synthetic_size
    gen_labels    = np.zeros(gen.shape[0])

    # data  = np.vstack(( data[:x], gen, data[x:]))
    # labels  = np.hstack((labels[:x], gen_labels , labels[x:]))
    data  = np.vstack((  gen, data))
    labels  = np.hstack(( gen_labels , labels))

    # data, labels  = self.shuffle_training_data_only(data, labels)

    
    return data, labels


  def loadSupervisedData(self):

    data_files = self.data_files
    data_count = len(data_files)

    training_size_equalized = np.inf
    total_training_size = 0

    

    for i in range(data_count):
      class_data, _ = self.loadfile(data_files[i], self.mask)
      training_size   = round(class_data.shape[0]*self.training_fraction)
      total_training_size+= training_size
      if training_size_equalized> training_size:
        training_size_equalized = training_size


    # holdout = round(total_training_size/10)
    training_size = training_size_equalized
    self.unique_class = set()


    for i in range(data_count):
      # if i == 0:
      #   training_size = holdout+training_size_equalized
      # else:
      #   training_size = training_size_equalized
      class_data, class_label = self.loadfile(data_files[i], self.mask)
      for l in class_label:
        self.unique_class.add(l)
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
    self.unique_class = sorted(self.unique_class)
    self.class_key = {idx:key for key,idx in enumerate( self.unique_class)}
    self.training_size = train_labels.shape[0]
    self.training_size_initial = self.training_size 
    
    if self.poison_persent>0:     
      # data, labels  = self.poison(data, labels)
      raise NotImplementedError

    
    if self.shuffle:
      # data, labels  = self.shuffle_training_data_only(data, labels)
      raise NotImplementedError


    scaler = MinMaxScaler()
    
    scaler.fit(train_data)

    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    if self.add_synthetic:
      # normalized_data, labels = self.add_synthetic_training_data(normalized_data, labels)
      raise NotImplementedError




    train_data    = train_data.astype(np.float64) 
    test_data     = test_data.astype(np.float64) 


    # pca = PCA()
    # data_1 = pca.fit_transform(train_data)
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of components')
    # plt.ylabel('Explained Variance')
    # plt.show()
    # pdb.set_trace() 
    pca = PCA(n_components=10)
    train_data = pca.fit_transform(train_data)

    test_data = pca.transform(test_data)



    return train_data, test_data, train_labels, test_labels


class FiveG_Dataset(Dataset):
  
  def __init__(self, mode, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, shuffle = False, noise_from_class = 1,  add_synthetic = False, guard = 0):
    self.mode = mode
    self.poison_persent   = poison_persent
    self.training_fraction  = training_fraction
    self.placement_percent  = placement_percent
    self.shuffle      = shuffle
    assert noise_from_class >=0 , 'class 0 must be clean data, cannot be noise'
    self.adversarial_class  = noise_from_class
    self.add_synthetic    = add_synthetic
    self.guard        = guard

    # self.get_val_data = False
    # self.get_test_data = False

    # pdb.set_trace()

    data_loader =   FiveG_data_loader(self.poison_persent, self.training_fraction, self.placement_percent, self.shuffle, self.adversarial_class, add_synthetic = self.add_synthetic, guard = self.guard)
    
    train_data, self.test_data, train_labels, self.test_labels      = data_loader.loadSupervisedData()
    self.train_data, self.val_data, self.train_labels,  self.val_labels = train_test_split(train_data, train_labels, test_size=0.1, random_state=42)
    self.class_key = data_loader.class_key
    self.class_map = {v: k for k, v in self.class_key.items()}
    self.partion_point = [key for key in self.class_key].index(0)

    

    # pdb.set_trace()

  # def set_val_data_mode(self):
  #   self.get_val_data = True
  #   self.get_test_data = False

  # def set_test_data_mode(self):
  #   self.get_val_data = False
    # self.get_test_data = True

  def __len__(self):
    if self.mode == 'test':
      return len(self.test_labels)
    elif self.mode == 'val':
      return len(self.val_labels)
    else:
      return len(self.train_labels)

  def __getitem__(self,idx):
    if self.mode == 'test':
      data =  self.test_data[idx]
      label = self.test_labels
    elif self.mode =='val':
      data =  self.val_data[idx]
      label = self.val_labels
    else:
      data =  self.train_data[idx]
      label = self.train_labels

    label1 = label[idx][0]
    label2 = label[idx][1]
    label_ =label[idx] 

    size = len(self.class_key)
    label_1 = np.eye(size, dtype='uint8')[self.class_key[label1]]
    label_2 = np.eye(size, dtype='uint8')[self.class_key[label2]]

    onehot_label = np.sum([label_1,label_2], axis=0)
    # print(self.class_map, label2)
    # pdb.set_trace()
    sample = {'data':data, 'labels': onehot_label.astype('f'), 'labels_dist': self.class_key[label1], 'labels_angle': self.class_key[label2]}  
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
      self.fc = nn.Linear(seq_length, num_classes) #fully connected last layer

      self.relu = nn.ReLU()
    
    def forward(self,x):
      x = torch.reshape(x.float(),   (x.shape[0], 1, x.shape[1]))
      h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
      c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
      # Propagate input through LSTM
      output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
      hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
      out = self.relu(hn)
      out = self.fc_1(out) #first Dense
      out = self.relu(out) #relu
      out = self.fc(out) #Final Output
      return out

class Experiment():
  def __init__(self):
    self.dataset=None
    # pdb.set_trace()
    # print(next(iter(train_loader)))   # 
    
    # resnet = models.resnet34(pretrained=True)
    # # list(resnet.children())[-3:]

    # model_wo_fc = nn.Sequential(*(list(resnet.children())[:-1]))

    # sample = next(iter(train_loader))
    # xx = sample['data']
    # xxx = torch.reshape(xx,   (xx.shape[0], 1, xx.shape[1]))
    # model = nn.LSTM(input_size=6, hidden_size=3, num_layers=1, batch_first=True) 
    # model = model.float()
    # out = model(xxx.float())

    # pdb.set_trace()

  def load_data(self):
    self.dataset = FiveG_Dataset('train')
    # train_set = self.dataset
    # self.dataset.set_val_data_mode()
    val_set = FiveG_Dataset('val')
    # self.dataset.set_test_data_mode()
    test_set = FiveG_Dataset('test')

    self.train_loader  = DataLoader(self.dataset, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    self.val_loader    = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    self.test_loader   = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, drop_last=True)
    self.num_classes = len(self.dataset.class_key) #number of output classes 
    self.class_map = self.dataset.class_map
    self.partion_point = self.dataset.partion_point
    # pdb.set_trace()

  def train_model(self, model, optimizer,  num_epochs=10, patience = 5):
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=lr)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    # criterion = lambda y_hat,y: F.cross_entropy(y_hat,y)
    min_val_loss= np.inf
    stagnant = 0

    for epoch in range(num_epochs):
      model.train()
      sum_loss = 0.0
      total = 0
      for sample in tqdm (self.train_loader, leave=False):
        # pdb.set_trace()
        x = sample['data']
        y = sample['labels']

        y_hat = model.forward(x) #forward pass
        optimizer.zero_grad() #caluclate the gradient, manually setting to 0
        # pdb.set_trace()
        # obtain the loss function
        loss = criterion(y_hat, y)
       
        loss.backward() #calculates the loss of the loss function     
        optimizer.step() #improve from loss, i.e backprop
        sum_loss += loss.item()*y.shape[0]
        total += y.shape[0]

      val_loss, val_acc, val_rmse = self.validation_metrics(model)

      if min_val_loss> val_loss:
        min_val_loss= val_loss
        stagnant = 0
      else:
        stagnant +=1
      if stagnant >= patience:
        break

      # if epoch % 5 == 0:
      print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))
      with open('Failed.py', 'a') as file:
        file.write(" %.4f, %.4f, %.4f, %.4f" % (sum_loss/total, val_loss, val_acc, val_rmse))

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
      y = sample['labels']
      y_d = sample['labels_dist']
      y_a = sample['labels_angle']
      y_hat = model.forward(x)
      loss = criterion(y_hat, y)
      y_d_hat = y_hat[:,:self.partion_point]
      y_a_hat = y_hat[:,self.partion_point:]
      # pdb.set_trace()
      pred_d = torch.max(y_d_hat, 1)[1]
      pred_a = torch.add(torch.max(y_a_hat, 1)[1],self.partion_point)
      correct += (pred_d == y_d).float().sum()
      correct += (pred_d == y_d).float().sum()
      total += 2*y.shape[0]
      sum_loss += loss.item()*y.shape[0]
      sum_rmse += np.sqrt(mean_squared_error(pred_d, y_d))*y.shape[0]
      sum_rmse += np.sqrt(mean_squared_error(pred_a, y_a))*y.shape[0]
    return sum_loss/(total/2), correct/total, sum_rmse/total

  def test_predict(self, model):
    model.eval()
    pred_A = None
    pred_D = None 
    Y_a = None 
    Y_d = None
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    # criterion = lambda outputs,y: F.cross_entropy(outputs,y)
    for sample in tqdm (self.test_loader, leave=False):
      # pdb.set_trace()
      x = sample['data']
      y = sample['labels']
      y_d = sample['labels_dist']
      y_a = sample['labels_angle']
      y_hat = model.forward(x)
      loss = criterion(y_hat, y)
      y_d_hat = y_hat[:,:self.partion_point]
      y_a_hat = y_hat[:,self.partion_point:]
      # pdb.set_trace()
      pred_d = torch.max(y_d_hat, 1)[1]
      pred_a = torch.add( torch.max(y_a_hat, 1)[1],self.partion_point) 
      pred_D = append_to(pred_D, pred_d)
      pred_A = append_to(pred_A, pred_a)
      Y_d = append_to(Y_d, y_d)
      Y_a = append_to(Y_a, y_a)
    return pred_A, pred_D, Y_a, Y_d

  def save_model(self, model, optimizer):
    torch.save({
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),}, 
      'saved_multiclass_model_PCA.pkl')

  def load_model(self, model, optimizer):
    checkpoint = torch.load('saved_multiclass_model_PCA.pkl')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


  def run(self, num_epochs = 100 ,learning_rate = 0.001, hidden_size = 2 ,num_lstm_layers = 1):

    
    if self.dataset == None:
      print('call load_data() before run()!!')
      return
      
    input_size = self.dataset.train_data.shape[1]

    model = LSTMnetwork(self.num_classes, input_size, hidden_size, num_lstm_layers) #our lstm class 
    model = model.float() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # model, optimizer = self.load_model(model, optimizer)

    self.train_model(model, optimizer, num_epochs)

    pred_A, pred_D, Y_a, Y_d = self.test_predict(model)

    distance_acc = accuracy_score(Y_d, pred_D)
    angle_acc = accuracy_score(Y_a, pred_A)

    print(f'distance accuracy score :{distance_acc:.3f}')
    print(f'angle accuracy score :{angle_acc:.3f}')

    # pdb.set_trace()

    label = [key for key in self.class_map]
    label_rename = [key for key in self.dataset.class_key]

    cm = confusion_matrix(Y_d, pred_D, labels=label[:self.partion_point])
    plot_CM(cm, [item*-1 for item in label_rename[:self.partion_point]])

    

    cm = confusion_matrix(Y_a, pred_A, labels=label[self.partion_point:])     
    plot_CM(cm, label_rename[self.partion_point:])

    self.save_model(model, optimizer)


    pdb.set_trace()

def plot_CM(cm, labels):
  import seaborn as sns
  import matplotlib.pyplot as plt     

  ax= plt.subplot()
  sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation

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