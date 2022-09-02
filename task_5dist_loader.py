import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from KitNET.KitNET import KitNET
NET = KitNET
# from KitNET.distNET import distNET
# NET = distNET
import pdb, traceback, csv
import math #   print('\x1bc')

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


import matplotlib, importlib
from matplotlib import pyplot as plt, colors

from math import dist
from sklearn.preprocessing import MinMaxScaler


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable 

def draw_roc_curve(roc_vals, show = True, tex_backend = False, lw=2):
	# print(tex_backend)
	if tex_backend:
		print('Useing Tex Backend..')

		try:
			matplotlib.use('TkAgg')
		except Exception as e:
			traceback.print_exc()
			pdb.set_trace()
		matplotlib.use('TkAgg')
		matplotlib.rcParams['figure.dpi'] = 400

		# importlib.reload(plt)
		plt.figure()
		plt.rcParams.update({
	    	"text.usetex": True,
	    	"font.family": "serif",
	    	"font.serif": ["Times"]})
	else:
		plt.figure()

    
	plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
	
	marker = iter(['+','o','v','^','<','>','x','.']*10)

	for roc_val, label_text in roc_vals:
		fpr 	=  roc_val[0] 
		tpr 	=  roc_val[1] 
		roc_auc =  roc_val[2] 
		plt.plot(fpr, tpr, marker = next(marker), label=f'AUC= {roc_auc:.4f} {label_text}' )

		lw = 1


	plt.xticks(np.arange(0.0, 1.1, step=0.1))
	plt.xlabel("False Positive Rate", fontsize=15)
	plt.yticks(np.arange(0.0, 1.1, step=0.1))
	plt.ylabel("True Positive Rate", fontsize=15)
	if tex_backend:
		plt.title( r'\textbf{ROC Curve Analysis For Each Jamming Scenario}', fontweight='bold', fontsize=15)
	else:
		plt.title( 'ROC Curve Analysis For Each Jamming Scenario', fontweight='bold')
	# plt.legend(prop={'size':13}, loc='lower right')
	plt.legend(prop={'size': 'medium','size': 12}, loc='lower right')
	plt.grid()
	plt.savefig('task_1/roc_unsupervized.png', bbox_inches='tight')#,dpi=200)




	# plt.xticks(np.arange(0.0, 1.1, step=0.1), fontsize=12)
	# plt.xlabel(r'False positive rate', fontsize=15)
	# plt.yticks(np.arange(0.0, 1.1, step=0.1), fontsize=12)
	# plt.ylabel(r'True positive rate', fontsize=15)
	# plt.title(r'\textbf{ROC curve analysis for LSTM detector}', fontweight='bold', fontsize=15)
	# plt.legend(prop={'size': 'medium'}, loc='lower right')
	# plt.grid()


	if show:
		plt.show()
	else:
		# plt.savefig('task_1/ROC_'+label_text+').png', bbox_inches='tight')
		plt.clf()

 
def save_fig(plot, filename):
	plot.savefig(filename, bbox_inches='tight')
	plot.close()


def plot_fig():
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator

	# noise 		= [0, 5, 5, 5, 5, 5, 10, 10, 10, 15, 15, 20, 20]
	# position 	= [0, 0.1, 0.25, 0.5, 0.75, 1, 0.1, 0.25, 0.5, 0.1, 0.25, 0.1, 0.25]
	# aoc 		= [1, 0.9819, 0.9485, 0.8845, 0.9189, 0.9992, 0.9775, 0.8048, 0.6817, 0.9755, 0.7564, 0.9755, 0.7046]

	aoc 		= [1,1,1,1,1, 0.951866667, 0.879766667, 0.7606, 0.842133333, 0.8808, 0.920366667, 0.736233333, 0.6961, 0.733733333, 0.8452, 0.9156, 0.7321, 0.69, 0.694233333, 0.790833333, 0.9165, 0.685933333, 0.673766667, 0.6765, 0.7018]

	noise 		= [0,5,10,15,20]
	# noise.reverse()
	position 	= [.1,.25,.5,.75,1]
	# position.reverse()
	# aoc = [[1,1,1,1,1],
	# 		[0.951866667, 0.879766667, 0.7606, 0.842133333, 0.8808],
	# 		[0.920366667, 0.736233333, 0.6961, 0.733733333, 0.8452],
	# 		[ 0.9156, 0.7321, 0.69, 0.694233333, 0.790833333],
	# 		[0.9165, 0.685933333, 0.673766667, 0.6765, 0.7018]]
	X, Y = np.meshgrid(noise, position)
	aoc = np.reshape(aoc, X.shape)
	print(aoc,'\n', X,'\n',Y)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.contour3D(noise, position, aoc)

	ax.set_xlabel('Noise percentage')
	ax.set_ylabel('Noise position')
	ax.set_zlabel('AOC average')

	plt.savefig('./'+'roc_unsupervized.png',dpi=200)
	plt.show()

class Task1_data_loader():
	"""docstring for Task1_data_loader"""
	def __init__(self, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, 
				shuffle = False, noise_from_class = 1, mask=[0,*range(9,32)],#,*range(7,32)1,12,13,40]
				add_synthetic=True):   # 13 is timestamp
		self.poison_persent 	= poison_persent
		self.training_fraction 	= training_fraction
		assert placement_percent >=0 and placement_percent <=1, 'placement_percent should be between 0(leftmost) and 1(rightmost)'
		self.placement_percent 	= placement_percent
		self.shuffle 			= shuffle
		
		data_folder = '5G'
		test_subfolder = '5G/testing_data'
		data_subfolder = '5G/training_data' 
		self.data_files = [	data_folder+"/SNR_data_10000_10feet.csv",  	# normal 
								# data_subfolder+"/SNR_data_5000__10feet_45degree.csv",
								# data_subfolder+"/SNR_data_5000__10feet_90degree.csv",
								# data_subfolder+"/SNR_data_5000__10feet_180degree.csv",
							# data_folder+"/SNR_data_10000_20feet.csv",
							data_subfolder+"/SNR_data_5000_20feet_0degree.csv",
						# test_subfolder+"/SNR_data_1000_25feet_0degree.csv",  		
								# data_subfolder+"/SNR_data_5000__20feet_45degree.csv",
								# data_subfolder+"/SNR_data_5000__20feet_90degree.csv",	
							# data_folder+"/SNR_data_10000_30feet.csv",
							data_subfolder+"/SNR_data_5000_30feet_0degree.csv", 
						# test_subfolder+"/SNR_data_1000_35feet_0degree.csv",    		
							# data_folder+"/SNR_data_10000_40feet.csv",
							data_subfolder+"/SNR_data_5000_40feet_0degree.csv", 		
							# data_folder+"/SNR_data_10000_50feet.csv",
							data_subfolder+"/SNR_data_5000_50feet_0degree.csv",
						test_subfolder+"/SNR_data_1000_25feet_0degree.csv",
						test_subfolder+"/SNR_data_1000_35feet_0degree.csv"
							]     
    	
		
		# self.data_files = self.data_files[:3] # masking out all but the first 4 classes
		assert noise_from_class >=0 , 'class 0 must be clean data, cannot be noise'
		assert noise_from_class <len(self.data_files) , 'class 0 must be clean data, cannot be noise'
		self.first = noise_from_class-1
		self.mask = mask
		self.add_synthetic = add_synthetic
		
	def loadfile(self, filename, mask = []):
		df  = pd.read_csv(filename, skiprows = 100)
		data = df.values
		inc = [17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29]
		rows, cols = data.shape
		import copy
		data_copy = copy.deepcopy(data)
		for r in range(rows):
			for c in range(cols):
				if c in inc:
					data[r,c]-=data_copy[r-1,c]
		data = np.delete(data,mask,1)
		return data

	def shuffle_training_data_only(self, data, labels):
		print('shuffle training data...')
		shuffel_point 	= self.benign_size
		train_data 		= data[:shuffel_point]
		eval_data 		= data[shuffel_point:]

		train_labels 	= labels[:shuffel_point]
		eval_labels 	= labels[shuffel_point:]

		train_data, train_labels = utils.shuffle(train_data, train_labels)

		# pdb.set_trace()
		data 	= np.vstack((train_data, eval_data ))
		labels 	= np.hstack((train_labels, eval_labels ))

		return data, labels

	def poison(self, data, labels): 	
		ts 		= self.training_size	 
		bs 		= self.benign_size	
		ps 		= round(ts*(self.poison_persent/100)) #ps = poison size
		x 		= min(round(self.training_size*self.placement_percent), ts-ps)	#self.placement_percent#
		x 		= max(round(self.training_size*.1),x)
		# post_poison_training_size = ps*self.placement_percent
		# x = ts-ps-post_poison_training_size
		# x = self.placement_percent

		# pdb.set_trace()

		
		data 	= np.vstack(( data[:x], data[bs:bs+ps], data[x:ts-ps], data[ts-ps:bs], data[bs+ps:]))
		labels 	= np.hstack((labels[:x], labels[bs:bs+ps], labels[x:ts-ps], labels[ts-ps:bs], labels[bs+ps:]))	
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
		gen_labels		= np.zeros(gen.shape[0])

		# data 	= np.vstack(( data[:x], gen, data[x:]))
		# labels 	= np.hstack((labels[:x], gen_labels , labels[x:]))
		data 	= np.vstack((  gen, data))
		labels 	= np.hstack(( gen_labels , labels))

		# data, labels 	= self.shuffle_training_data_only(data, labels)



		
		return data, labels


	def rescale(self, data):
		# from sklearn.metrics.pairwise import rbf_kernel
		# k = rbf_kernel(data,data[:50])
		# from sklearn.manifold import TSNE
		# tsne = TSNE(n_components=3, n_iter=500)
		# k = pd.DataFrame(tsne.fit_transform(data))
		# k = k.to_numpy()
		# k 			= np.hstack((data,k))
		# pdb.set_trace()
		k = np.square(data)
		return k



	def loadData(self):
		norm  = self.loadfile(self.data_files[0], 	self.mask)

		jaming_data_files = self.data_files[1:]
		jaming_data_count = len(jaming_data_files)

		# print(self.first, )

		jamming_data =[]
		data_files = [self.data_files[0]]
		for i in range(self.first, self.first+jaming_data_count):	
			# print(i, jaming_data_files[i%jaming_data_count])
			jamming_data.append(self.loadfile(jaming_data_files[i%jaming_data_count], self.mask))
			data_files.append(jaming_data_files[i%jaming_data_count])

		self.data_files = data_files

		self.training_size 	= round(norm.shape[0]*self.training_fraction)
		self.training_size_initial = self.training_size
		self.benign_size 	= norm.shape[0]
		
		data =norm

		for i, data_i in enumerate(jamming_data):
			data 			= np.vstack((data,data_i))

		labels		= np.zeros(norm.shape[0])
		for i, data_i in enumerate(jamming_data):
			data_label	 	=  np.ones(data_i.shape[0])*(i+1)
			labels 			= np.hstack((labels,data_label))

		
		if self.poison_persent>0:			
			data, labels 	= self.poison(data, labels)

		
		if self.shuffle:
			data, labels 	= self.shuffle_training_data_only(data, labels)


		scaler = MinMaxScaler()
		normTrain = norm[:self.training_size]
		scaler.fit(normTrain)

		normalized_data = scaler.transform(data)

		if self.add_synthetic:
			normalized_data, labels = self.add_synthetic_training_data(normalized_data, labels)




		data 		= normalized_data.astype(np.float64)

		# data 		= self.rescale(data)
		# pca = PCA()
		# data_1 = pca.fit_transform(data)
		# plt.figure()
		# plt.plot(np.cumsum(pca.explained_variance_ratio_))
		# plt.xlabel('Number of components')
		# plt.ylabel('Explained Variance')
		# plt.show()
		# pdb.set_trace()	
		# pca = PCA(n_components=4)
		# data = pca.fit_transform(data)

		data_size = len(data)
		self.test_size = data_size-2*(899) 

		return data, labels 


def remap_y(y):
	for i in range(len(y)):
		if y[i]<=4:
			y[i]+=1
		elif y[i] == 5:
			y[i]=2.5
		else:
			y[i]=3.5
	return y		

class Task1():
	"""docstring for Task1"""
	def __init__(self, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, shuffle = False, 
				noise_from_class = 1, average_model= True, window_size= 500, show_training_variance = False, draw_hist = False, add_synthetic = False):
		# super(Task1, self).__init__()
		self.poison_persent 	= poison_persent
		self.training_fraction 	= training_fraction
		self.placement_percent 	= placement_percent
		self.shuffle 			= shuffle
		self.training_variance 	= show_training_variance
		self.average_model 		= average_model
		assert noise_from_class >=0 , 'class 0 must be clean data, cannot be noise'
		self.adversarial_class	= noise_from_class
		self.training_consistancy_std = 0
		self.training_consistancy_mean = 0
		self.draw_hist 			= draw_hist
		self.window_size 		= window_size
		self.add_synthetic 		= add_synthetic
		self.tanh_flag = False

	def loadData(self):

		data_loader =  	Task1_data_loader(self.poison_persent, self.training_fraction, self.placement_percent, self.shuffle, self.adversarial_class, add_synthetic = self.add_synthetic)
		data, labels  		= data_loader.loadData()
		self.data_files 	= data_loader.data_files
		self.training_size 	= data_loader.training_size	
		self.test_size 		= data_loader.test_size
		self.training_size_initial  = data_loader.training_size_initial	
		self.data 		= data
		self.labels 	= labels

		print(f'{self.training_size=}')

		data_shape 				= self.data.shape
		self.num_samples 		= data_shape[0]
		self.num_features 		= data_shape[1]

		# pdb.set_trace()

		# results = tanRMSEs[training_size+1:]
		# gold 	= labels[training_size+1:]

		return self.data, self.labels 

	def train_and_predict(self):
		
		max_autoencoder_size	= 10
		FM_grace_period			= round(self.training_size_initial/10)
		AD_grace_period			= self.training_size - FM_grace_period
		learning_rate			= 0.1
		hidden_ratio			= 0.75

		AnomDetector = NET(self.num_features,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

		# ccc = np.zeros(num_features)
		# x = normalized_data[0]
		# for i in range(len(ccc)):
		# 	print(ccc[i],type(ccc[i]),x[i],type(x[i]))

		# pdb.set_trace()

		# msgLast = ''
		window_size = self.window_size
		# if self.training_variance:
		std_window_size = FM_grace_period
		self.STD = []
		self.Mean = []

		RMSEs = []
		fixed_RMSEs  = []
		# pbar = tqdm(total=num_samples, leave=True)
		for i in tqdm(range(self.num_samples), leave=False):
			# data_shape
			
			try:
				# process KitNET
				rmse, msg = AnomDetector.process(self.data[i])  # will train during the grace periods, then execute on all the rest.


				
				# if i> self.training_size:
				# 	rmse *=-1

				# if msgLast!= msg:
				# 	msgLast = msg
				# 	pbar.set_description(msg)

				RMSEs.append(rmse)

				if i> FM_grace_period+std_window_size and i<self.training_size: # self.training_variance and
					past_training_window = RMSEs[min(i-std_window_size+1,self.training_size):i+1]

					mean = np.mean(np.tan(past_training_window))
					std = np.std(np.tan(past_training_window))
					# pdb.set_trace()
					self.STD.append(std)
					self.Mean.append(mean)
					


				# if i == self.training_size:					
				# 	self.results_raw 	= np.tanh(RMSEs)
				# 	colors_set = ['green','red','blue','purple']
				# 	plt.scatter(range(len(self.results_raw)),self.results_raw,s=1, marker='.',c=self.labels[:len(self.results_raw)], cmap=colors.ListedColormap(colors_set) )#label='b')
				# 	# plt.scatter(range(len(self.results_raw)),self.results_raw,s=1, marker='.',c='g')
				# 	plt.scatter(range(FM_grace_period, FM_grace_period+len(self.STD)), self.STD, s=.1, marker='.',c='b')
				# 	plt.show()
				# 	pdb.set_trace()
				# 	# return

				
				

				if self.average_model and i>self.training_size:
					actual_rmse = rmse 
					rmse = np.mean(np.tan( RMSEs[max(i-window_size+1,0):i+1]) )
					# if i%100 ==0:
					# 	print(i, actual_rmse, '>', i-window_size+1, ':',i+1, rmse)
					
				fixed_RMSEs.append(rmse)

			except Exception as e:
				traceback.print_exc()
				pdb.set_trace()

			# if i%100==0:
				# pbar.set_description(msg)
				# pbar.update(100)
				# pdb.set_trace()
			# if i==600:
			# 	pdb.set_trace()

		self.clusters = AnomDetector.v

		if self.training_variance:
			window = std_window_size*2
			self.training_consistancy_std =  np.mean(self.STD[-window:])/np.mean(self.STD[window:]) 
			self.training_consistancy_mean =  np.mean(self.Mean[-window:])/np.mean(self.Mean[window:])
			
			print(f'Training std consistancy :{self.training_consistancy_std:.4f}')
			print(f'Training mean consistancy :{self.training_consistancy_mean:.4f}')


		# tanRMSEs =  np.tanh(fixed_RMSEs)

		# results = tanRMSEs[training_size+1:]
		# gold 	= self.labels[training_size+1:]
		
		if self.tanh_flag:
			self.results_raw 	= np.tanh(RMSEs)
			self.results 		= np.tanh(fixed_RMSEs)
		else:
			self.results_raw 	= RMSEs#np.tanh(RMSEs)
			self.results 		= fixed_RMSEs#np.tanh(fixed_RMSEs)
		self.gold 			= np.round(np.tanh(self.labels))

		

		# results =  tanRMSEs
		# RMSEData = np.vstack((RMSEs,tanRMSEs))
		# df = pd.DataFrame(RMSEData)

		return self.gold, self.results

	def plot_results(self, threshold = None, fig = True): #threshold set by inspection
		# window_size = self.window_size
		if not threshold:
			# if self.training_variance:
			# if self.average_model:
			self.threshold = np.mean(self.Mean)
			# else:
				# self.threshold = np.mean(self.Mean)+np.mean(self.STD)
			# else:
			# 	threshold_vals = self.results[self.training_size-window_size:self.training_size]
			# 	std = np.std(threshold_vals)
			# 	mean = np.mean(threshold_vals)
			# 	threshold = mean+1*std
		else:
			self.threshold = threshold
			
		# print(threshold)

		pred = []
		for result in self.results:
			if result>self.threshold:
				pred.append(1)
			else:
				pred.append(0)

		
		acc = accuracy_score(self.gold[self.training_size:], pred[self.training_size:])
		print(f'accuracy score :{acc:.3f}')

		idx = []
		start_at = self.training_size+1
		idx.append([0,start_at])		
		num_classes = int(np.max(self.labels))+1
		for i in range(num_classes-1):
			end_at = np.where(self.labels[self.training_size+1:]==i+1)[0][0]+self.training_size+1
			idx.append([start_at,end_at])
			start_at = end_at
		idx.append([start_at,-1])

		print('class wise accuracy:')
		for i in range(1,len(idx)):
			acc_i = accuracy_score(self.gold[idx[i][0]:idx[i][1]], pred[idx[i][0]:idx[i][1]])
			# rec_i = recall_score(self.gold[idx[i][0]:idx[i][1]], pred[idx[i][0]:idx[i][1]])
			# pre_i = precision_score(self.gold[idx[i][0]:idx[i][1]], pred[idx[i][0]:idx[i][1]])
			if math.isnan(acc_i):
				pdb.set_trace()

			print(f'\t{self.data_files[i-1][19:-4]}  :\t\t{acc_i:.3f}')#\t{pre_i:.3f}\t{rec_i:.3f}')


		
		# pdb.set_trace()

		
		colors_set = ['green','red','blue','purple', 'yellow','cyan','brown','orange','green','red','blue','purple', 'yellow','cyan','brown','orange']
		colors_set = colors_set[:len(self.data_files)]
		# classes = ['no_interference','nr_11dBm','lte_ul_11dBm','lte_dl_11dBm']

		try:
			plt.scatter(range(len(self.results_raw)),self.results_raw,s=1, marker='.',c=self.labels, cmap=colors.ListedColormap(colors_set) )#label='b')
			if self.average_model:
				plt.scatter(range(self.training_size,self.num_samples), self.results[self.training_size:], s=.1, marker='.',c='k')


			# self.plotFeatureMean = True
			# if self.plotFeatureMean:
			# 	feature_mean = np.mean(self.data,1)
			# 	plt.scatter(range(len(self.results_raw)), feature_mean-.1,s=.1, marker='.',c='m')

			if self.training_variance:
				FM_grace_period			= round(self.training_size/10)
				plt.scatter(range(FM_grace_period, FM_grace_period+len(self.STD)), self.STD, s=.1, marker='.',c='y')
		except Exception as e:
			traceback.print_exc()
			pdb.set_trace()

		for i, file in enumerate(self.data_files):	
			try:
				classname= file[18:-4]
			except Exception as e:
				classname = file	
			plt.scatter([-1],[-1],s=3, marker='o', color=colors_set[i] , label=classname)
			# print(i,' ',file[7:],' ',colors_set[i])
		if self.average_model:
			plt.scatter([-1],[-1],s=3, marker='o', color='k' , label='running average')


		if self.tanh_flag:
			plt.ylim((-0.01,1.01))
		plt.xlim(( 0, self.num_samples ))
		plt.axhline(y=self.threshold, xmin= self.training_size/self.num_samples, color='r', ls='--')

		# plt.axvline(x=596, ymin=-0.1, ymax=1.1, color='k')
		plt.axvspan(0, self.training_size, facecolor='g', alpha=0.1)
		plt.text(self.training_size-350, -0.05, 'Train', fontsize=8)
		plt.text(self.training_size+100, -0.05, 'Test', fontsize=8)
		# plt.axvline(x=0, ymin=0, ymax=1, color='k')

		# plt.axvline(x=bloc_start-training_size, ymin=0, ymax=0.05, color='k')
		# plt.axvline(x=bloc1_start-training_size, ymin=0, ymax=0.05, color='k')
		# plt.axvline(x=bloc2_start-training_size, ymin=0, ymax=0.05, color='k')
		# # plt.axvline(x=bloc2_end-training_size, ymin=0, ymax=0.05, color='k')
		# unique_classes = list(set(classes))
		plt.legend()


		# x_tick_points = [bloc_start-1.9*training_size, bloc1_start-1.9*training_size, bloc2_start-1.9*training_size, bloc2_end-1.9*training_size]
		# xticks = ['no_interference','nr_11dBm','lte_ul_11dBm','lte_dl_11dBm']
		# plt.xticks(x_tick_points, xticks)
		# plt.gca().axes.xaxis.set_ticklabels([])

		

		plt.xticks([], [])
		if fig:
			plt.show()
			# pdb.set_trace()
		else:
			try:
				plt.savefig('task_1/poison_'+str(self.poison_persent)+'('+str(self.placement_percent)+').png', bbox_inches='tight')
				plt.close()
			except Exception as e:
				traceback.print_exc()
				pdb.set_trace()
			
		# 
		# pdb.set_trace()

	def draw_histograms(self, buckets =100, uniform_y_axis = True):
		hist_idx = []
		start_at = self.training_size+1
		num_classes = int(np.max(self.labels))+1
		fig, axs = plt.subplots(num_classes)
		colors_set = ['green','red','blue','purple']
		# try:
		for i in range(1,num_classes-1):
			end_at = np.where(self.labels==i+1)[0][0]
			hist_idx.append([start_at,end_at])
			start_at = end_at
		i +=1
		hist_idx.append([start_at,-1])
		for i in range(num_classes):
			axs[i].hist(self.results_raw[hist_idx[i][0]:hist_idx[i][1]], buckets, color = colors_set[i])
			# hist = plt.hist(self.results_raw[hist_idx[i][0]:hist_idx[i][1]], buckets, color = colors_set[i])
			# file = self.data_files[i]
			# save_fig(plt, 'task_1/HIST_'+file[7:-4]+'.png',)

		if uniform_y_axis:
			plt.setp(axs, ylim=axs[-1].get_ylim(), xlim=axs[-1].get_xlim())
		else:
			plt.setp(axs, xlim=axs[-1].get_xlim())

		plt.show()

		pred = []
		for result_raw in self.results_raw:
			if result_raw>self.threshold:
				pred.append(1)
			else:
				pred.append(0)

		pdb.set_trace()

		for i in range(num_classes):
			file = self.data_files[i]
			acc = accuracy_score(self.gold[hist_idx[i][0]:hist_idx[i][1]], pred[hist_idx[i][0]:hist_idx[i][1]])
			print(f'accuracy score for {file[6:-4]:s} :{acc:.3f}')
		acc = accuracy_score(self.gold[hist_idx[0][0]:hist_idx[-1][1]], pred[hist_idx[0][0]:hist_idx[-1][1]])
		print(f'overall accuracy :{acc:.3f}')

		# top=0.985,
		# 	bottom=0.039,
		# 	left=0.022,
		# 	right=0.117,
		# 	hspace=0.188,
		# 	wspace=0.2

		# except Exception as e:
		# 	traceback.print_exc()
		# 	pdb.set_trace()
	
	def calc_roc(self, ):
		gold = self.gold[self.training_size:] 
		results = self.results[self.training_size:]

		fpr, tpr, thresholds = roc_curve(gold, results, pos_label=1, drop_intermediate=True)
		roc_auc = auc(fpr, tpr)

		
		print(f'Auc :{roc_auc:.13f}')

		# pdb.set_trace()

		return [fpr, tpr, roc_auc]

	def filter_Outlier(self, data, label):
		mean 	= np.mean(data)
		std 	= 0.5#np.std(data)
		x = []
		y = []
		thres_min = mean-std
		thres_max = mean+std
		for i in range(len(data)):
			if data[i]<=thres_max and data[i]>=thres_min:
				x.append(data[i])
				y.append(label[i])
		# print(label[0], mean,np.std(data))
		# x.append(mean)
		# y.append(label[0])
		return np.array(x),np.array(y)


	def predict(fit, X):
		Y = []
		for x in X:
			if x == 0:
				y=1
			else:
				y = fit[1] - fit[0] * np.log(x)
			Y.append(y)
		return np.array(Y)

	def test_regression(self):
		try:
			X 	= np.array(self.results).reshape(-1,1)
			Y 	= remap_y(self.labels)
			X_train = X[self.training_size:self.test_size]
			Y_train = Y[self.training_size:self.test_size]
			X_test = X[self.test_size:]
			Y_test = Y[self.test_size:]

			class_idx = []
			start_at = 0
			num_classes = int(np.max(self.labels))+1
			for i in range(1,num_classes-1):
				end_at = np.where(self.labels==i+1)[0][0] -self.training_size
				class_idx.append([start_at,end_at])
				start_at = end_at
			
			class_idx.append([start_at,-1])
			data = None
			label = []
			for idx in class_idx:
				d = X_train[idx[0]:idx[1]]
				l = Y_train[idx[0]:idx[1]]
				D,L = self.filter_Outlier(d,l)
				if data is None:
					data 	= D
					label 	= L
				else:
					data = np.vstack((data,D))
					label = np.hstack((label,L))
				print(idx, d.shape,D.shape)

			colors_set = ['green','red','blue','purple', 'yellow','cyan','brown','orange','green','red','blue','purple', 'yellow','cyan','brown','orange']
			plt.scatter(range(len(data)),data,s=1, marker='.',c=label, cmap=colors.ListedColormap(colors_set) )
			plt.show()
			pdb.set_trace()

			model = LinearRegression()

			neigh = KNeighborsRegressor(n_neighbors=10)
			svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
			# svr_lin = SVR(kernel="linear", C=100, gamma="auto")
			# svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

			
			model.fit(data,label)
			svr_rbf.fit(data,label)
			# svr_lin.fit(data,label)
			# svr_poly.fit(data,label)
			neigh.fit(data,label)
			fit = np.polyfit(np.log(data)[:,0], label, 1)


			# Y_pred = model.predict(X_test)
			Y_l = model.predict(X)
			Y_s =svr_rbf.predict(X)
			# Y_s2 =svr_lin.predict(X)
			# Y_s3 =svr_poly.predict(X)
			Y_k =neigh.predict(X)
			# Y_tr =neigh.predict(X_train)
			# Y_t =neigh.predict(X_test)
			Y_log = predict(fit, X)
			
			print(fit)

			plt.ion()

			plt.plot(X,Y,'g.')
			plt.plot(X,Y_l,'r.')
			plt.plot(X,Y_s,'b.')
			# plt.plot(X,Y_s2,'b.')			
			# plt.plot(X,Y_s3,'b.')
			plt.plot(X,Y_k,'y.')
			plt.plot(X,Y_log,'k.')
			# plt.plot(X_train,Y_tr,'y.')
			# plt.plot(X_test,Y_t,'r.')

			plt.show()
			pdb.set_trace()



		except Exception as e:
			traceback.print_exc()
			pdb.set_trace()

	def eval(self, threshold, fig, roc):
		self.loadData()
		self.train_and_predict()
		
		self.plot_results(threshold, fig)
		
		self.test_regression()

		pdb.set_trace()

		if self.draw_hist:
			self.draw_histograms(500, False)
			# self.draw_histograms(500)
			# pdb.set_trace()


		roc_val = self.calc_roc()
		if roc:
			draw_roc_curve([[roc_val, 'ROC curve']] )


		# feature_names = ['dl_bitrate_0', 'ul_bitrate_0', 'dl_tx_0', 'ul_tx_0', 'dl_retx_0', 'ul_retx_0', 'pusch_snr_0', 'epre_0', 'cqi_0', 'ri_0',#'time', 
		# 	'ul_phr_0', 'ul_path_loss_0', 'dl_mcs_0', 'ul_mcs_0', 'turbo_decoder_min_0', 'turbo_decoder_avg_0', 'turbo_decoder_max_0', 'ue_index', 
		# 	'max_ue_index', 'dl_bitrate_1', 'ul_bitrate_1', 'dl_tx_1', 'ul_tx_1', 'dl_retx_1', 'ul_retx_1', 'pusch_snr_1', 'epre_1', 'cqi_1', 'ri_1', 
		# 	'ul_phr_1', 'ul_path_loss_1', 'dl_mcs_1', 'ul_mcs_1', 'turbo_decoder_min_1', 'turbo_decoder_avg_1', 'turbo_decoder_max_1'
		# 	]
		# print('The features clustered as :')
		# for c in self.clusters:
		# 	print('---------------')
		# 	for f in c:
		# 		print(feature_names[f])

		# pdb.set_trace()
		return roc_val, self.training_consistancy_std, self.training_consistancy_mean



def run_all(multi_position =False, use_tex_backend = False, noise_from_class = 1, average_model= True, window_size = 500, show_training_variance = False, fig=True):

	roc_Vals 	= []

	task = Task1(training_fraction = 0.6, shuffle=False, noise_from_class=noise_from_class, average_model=average_model, show_training_variance=show_training_variance)
	# task = Task1(poison_persent = 5, training_fraction = 0.6, placement_percent = .5, shuffle=False)
	roc_val, Tstd, Tmean = task.eval(threshold = None, fig=fig, roc=False)
	roc_Vals.append([roc_val, 'No poisoning'])

	with open('task_1/AUC_agg.csv', 'w', newline='') as f:
		writer = csv.writer(f)

		writer.writerow(['Noise', 'Position', 'AUC',  'Training std', 'Training mean'])
		writer.writerow([0,0,roc_val[2], Tstd, Tmean])



	noises = [5, 10,  15, 20, 25, 30]
	if multi_position:
		starting_position = [[.1, .25, .5, .75, 1]]* len(noises)#,[.1, .25, .5 ],[.1, .25],[.1, .25]]
	else:
		starting_position = [[.25]] * len(noises)

	for idx, i in enumerate(noises):
		# print(idx,i)
		try:
			for j in starting_position[idx]:
				# task = Task1(poison_persent = 5.6, training_fraction = 0.9, placement_percent = 25, shuffle=False)
				task = Task1(poison_persent = i, training_fraction = 0.6, placement_percent = j, shuffle=False, noise_from_class=noise_from_class, average_model=average_model, window_size= window_size, show_training_variance=show_training_variance)
				roc_val,  Tstd, Tmean= task.eval(threshold = None, fig=fig, roc=False)

				with open('task_1/AUC_agg.csv', 'a', newline='') as f:
					writer = csv.writer(f)
					writer.writerow([i,j,roc_val[2], Tstd, Tmean])

				if use_tex_backend:
					roc_Vals.append([roc_val, 'Poison \% = '+str(i)]) #+' at '+str(j)
				else:
					roc_Vals.append([roc_val, 'Poison % = '+str(i)]) #+' at '+str(j)
				# draw_roc_curve(roc_Vals[-len(strating_position[idx]):], show=False, lw=1)
		except IndexError as e:
				pass
		
	# print(use_tex_backend)
	try:
		draw_roc_curve(roc_Vals, tex_backend= use_tex_backend)
	except Exception as e:
		traceback.print_exc()
		pdb.set_trace()
	
	

	# pdb.set_trace()


def explore_data(training_fraction = 0.6, placement_percent = 0.1, poison_persent = 5, noise_from_class = 1,  shuffle = False, 
	plot_jamming_classes = False, plot_stats= True, TSNE = False, use_gen =False):
	# matplotlib.use('TkAgg')

	try:
		# task1 = Task1_data_loader(mask = [0,1,12,13,40])  # 13 is timestamp
		task1 = Task1(poison_persent, training_fraction , placement_percent, shuffle, noise_from_class)
		data, labels = task1.loadData()

		feature_names = ['dl_bitrate_0', 'ul_bitrate_0', 'dl_tx_0', 'ul_tx_0', 'dl_retx_0', 'ul_retx_0', 'pusch_snr_0', 'epre_0', 'cqi_0', 'ri_0',#'time', 
			'ul_phr_0', 'ul_path_loss_0', 'dl_mcs_0', 'ul_mcs_0', 'turbo_decoder_min_0', 'turbo_decoder_avg_0', 'turbo_decoder_max_0', 'ue_index', 
			'max_ue_index', 'dl_bitrate_1', 'ul_bitrate_1', 'dl_tx_1', 'ul_tx_1', 'dl_retx_1', 'ul_retx_1', 'pusch_snr_1', 'epre_1', 'cqi_1', 'ri_1', 
			'ul_phr_1', 'ul_path_loss_1', 'dl_mcs_1', 'ul_mcs_1', 'turbo_decoder_min_1', 'turbo_decoder_avg_1', 'turbo_decoder_max_1'
			]


		hist_idx = []
		start_at = task1.training_size+1
		hist_idx.append([0,start_at])		
		num_classes = int(np.max(labels))+1
		for i in range(num_classes-1):
			end_at = np.where(labels==i+1)[0][0]
			hist_idx.append([start_at,end_at])
			start_at = end_at
		i +=1

		hist_idx.append([start_at,-1])

		num_features = data.shape[1]
		# pdb.set_trace()

		if plot_stats:

			# mask_stable = [5,14,17,18,28,31,6,7,9,13]
			# data = np.delete(data,mask_stable,1)
			# feature_names = np.delete(feature_names,mask_stable)

			num_features =data.shape[1]

			normTrain = data[hist_idx[0][0]:hist_idx[0][1]]
			normTest = data[hist_idx[1][0]:hist_idx[1][1]]

			# pdb.set_trace()

			normTrain_mean = np.mean(normTrain,0)
			normTest_mean = np.mean(normTest,0)
			normTrain_std = np.std(normTrain,0)
			normTest_std = np.std(normTest,0)
			# dist_i = dist(normTrain_mean, normTest_mean)
			
			# plt.plot(range(len(normTrain_mean)), normTrain_mean - normTest_mean, label=name)		
			# print(name,':',dist_i)

			# colours = ['cyan','green','red','blue','purple','yellow']
			colours = ['cyan','green','red','blue','purple','brown','orange','dodgerblue','palegreen','yellow']


			xvals = [*range(num_features)]

			# for sample in [normTrain_mean]: 
			# plt.plot(xvals, normTrain_mean, marker='.', color= colours[0])#, label='training')
			plt.errorbar(xvals, normTrain_mean, yerr=normTrain_std, fmt="o", color= colours[0], label='training')
			# xvals1 =  [x+ (poison_persent/100) for x in xvals]
			# plt.errorbar(xvals1, normTrain_mean, yerr=normTrain_std, fmt="o", color= str(max(3*poison_persent/100,0)), label='training')
			# plt.errorbar(range(num_features), normTrain_mean, yerr=c, fmt="o")
			name = task1.data_files[0]
			name = name[19:-4]
			# for sample in [normTest_mean]:
			xvals =  [x+ .05 + (poison_persent/100) for x in xvals]
			# plt.plot(xvals, normTest_mean, marker='.', color= colours[1], label=name)
			plt.errorbar(xvals, normTest_mean, yerr=normTest_std, fmt="o", color= colours[1], label=name)

			if plot_jamming_classes:			

				for i in range(1,num_classes):
					# data = scaler.transform(data)
					data_i = data[hist_idx[i][0]:hist_idx[i][1]]
					
					jammin_mean= np.mean(data_i,0)
					jammin_std= np.std(data_i,0)
					dist_i = dist(normTrain_mean, jammin_mean)
					name = task1.data_files[i]
					name = name[7:-4]
					# for sample in [jammin_mean]:
					xvals =  [x+ ((i/50)+.05) for x in xvals] # shift plot to right
					# plt.plot(xvals, jammin_mean, marker='.', color= colours[2+i])
					plt.errorbar(xvals, jammin_mean, yerr=jammin_std, fmt="o", color= colours[1+i], label=name)
				# 	plt.plot(range(len(normTrain_mean)), normTrain_mean - jammin_mean, label=name)
				# 	print(name,':',dist_i)

			gen_df = pd.read_csv('task_1/GEN.csv')
			gen =  gen_df.values
			# print(gen.shape)

			gen_mean = np.mean(gen,0)
			gen_std = np.std(gen,0)
			xvals =  [x+ ((i/50)+.05)for x in xvals]
			plt.errorbar(xvals, gen_mean, yerr=gen_std, fmt="o", color= colours[-1], label='generated')



			
			# pdb.set_trace()


			x_ticks = feature_names
			plt.xticks([*range(num_features)], x_ticks, rotation=45, ha='right', rotation_mode='anchor')#, fontsize=25)
			# plt.xticks(np.arange(0.0, num_features, step=1), fontsize=25)
			plt.xlabel("Features")#, fontsize=25)
			# plt.yticks(np.arange(0.0, 1.1, step=0.1))
			plt.ylabel("Mean feature values (scaled)")#, fontsize=25)
			plt.grid()
			plt.legend()#fontsize=25)
			plt.savefig('task_1/data_trend___.png', pad_inches=15,dpi=2000)
			plt.show()
		else:
			if TSNE:
				# colours = ['palegreen','green','navy','aqua','firebrick','cyan','darkred','darkturquoise','maroon','yellow']
				colours = ['green','green','navy','aqua','yellow','aqua','yellow','aqua','yellow','palegreen']

				from sklearn.manifold import TSNE		
				if use_gen:
					gen_df = pd.read_csv('task_1/GEN.csv')
					gen =  gen_df.values
					print(gen.shape)

					gen_mean = np.mean(gen,0)
					gen_std = np.std(gen,0)
					# pdb.set_trace()
					hist_idx[-1][1]= data.shape[0]
					data = np.vstack((data,gen))
					hist_idx.append([hist_idx[-1][1],data.shape[0]])
				print("Please wait, Computing TSNE ...")
				tsne = TSNE(n_components=2, n_iter=500)
				tsne_results = pd.DataFrame(tsne.fit_transform(data))
				# pdb.set_trace()

				# if use_gen:
				# 	plt.scatter(tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],0], tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],1],marker='.', c=colours[j+1], label='generated')

				for j in reversed(range(num_classes)):
					name = task1.data_files[j]
					name = name[19:-4]
					# print(j)
					plt.scatter(tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],0], tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],1],marker='.', c=colours[j+1], label=name)
				j=j+1
				print(j)

				plt.scatter(tsne_results.iloc[:hist_idx[0][1],0], tsne_results.iloc[:hist_idx[0][1],1], marker='.', c=colours[0], label='Train') #alpha=0.2


				plt.legend()
				plt.show()
				pdb.set_trace()
			else :
				if plot_jamming_classes:
					norm = data[hist_idx[0][0]:]
				else:
					norm = data[hist_idx[0][0]:hist_idx[1][1]]
				idx = [*range(norm.shape[0])]

				# mask_stable = [5,14,17,18,28,31,6,7,9,13]
				# norm = np.delete(norm,mask_stable,1)
				# feature_names = np.delete(feature_names,mask_stable)
				num_features =norm.shape[1]

				for i in range(num_features):
					sample = norm[:,i]

					plt.plot(idx, sample+3*i,'-')


				# plt.plot([hist_idx[0][1],hist_idx[0][1]], [-1, 3*i],'k')

				# print(hist_idx)

				# position = hist_idx[3][1]+346

				# plt.plot([position,position], [-1, 3*i],'r')


				if plot_jamming_classes:
					for j in range(1, num_classes):
						plt.plot([hist_idx[j][1],hist_idx[j][1]], [-1, 3*i],'k')



				plt.yticks([3*i for i in range(num_features)],feature_names)#, fontsize=25)
				plt.show()
				# pdb.set_trace()


				# plt.ion()
				# plt.show()
			# PCA
			# from sklearn.decomposition import PCA			
			# pca = PCA(n_components=2)

			# pca.fit(data[:hist_idx[0][1]])
			# pca_train = pd.DataFrame(pca.transform(data[:hist_idx[0][1]]))
			# plt.scatter(pca_train.iloc[:, 0].values, pca_train.iloc[:,1].values, alpha=0.2, c=colours[0], label='Train')

			# for j in range(num_classes):
			# 	name = task1.data_files[j]
			# 	name = name[7:-4]
			# 	pca_j = pd.DataFrame(pca.transform(data[hist_idx[j][1]:hist_idx[j+1][1]]))
			# 	plt.scatter(pca_j.iloc[:, 0].values, pca_j.iloc[:,1].values,  alpha=0.2, c=colours[j+1], label=name)
			# plt.legend()
			# plt.show()

	except Exception as e:
		traceback.print_exc()
		pdb.set_trace()
	# pdb.set_trace()

def run(poison_persent, training_fraction, placement_percent, shuffle=False, noise_from_class = 1, average_model= True, window_size= 500, show_training_variance = False, fig=True, roc =True, draw_hist =False, add_synthetic = False):

	roc_Vals 	= []

	# task = Task1(training_fraction = 0.6, shuffle=False)
	task = Task1(poison_persent, training_fraction , placement_percent, shuffle, noise_from_class, average_model, window_size, show_training_variance, draw_hist, add_synthetic)

	roc_val, Tstd, Tmean  = task.eval(threshold = None, fig=fig, roc=False)


	if poison_persent> 0:
		try:			
			print('poisoned training data from',task.data_files[task.adversarial_class])
		except Exception as e:
			pdb.set_trace()

	if roc:
		message = 'poison:'+str(poison_persent)+' placement:'+ str(placement_percent)
		roc_Vals.append([roc_val, message])	

		draw_roc_curve(roc_Vals)

	# import pickle
	# with open('saved_data.pkl', 'wb') as outp:
	# 	pickle.dump(task, outp, pickle.HIGHEST_PROTOCOL)
	# 	print('saved !')



	# pdb.set_trace()

def main():
	print('\x1bc')
	use_tex_backend = False
	if use_tex_backend:
		matplotlib.use('TkAgg')
	# plt.ion()
	# explore_data(poison_persent = 0, training_fraction = 0.66, placement_percent = .5, plot_jamming_classes = True, plot_stats= True, TSNE = False, use_gen = False)
	run(poison_persent = 0, training_fraction = 0.6, placement_percent = .1, shuffle=False, noise_from_class = 1, 
		average_model= False, window_size= 100, show_training_variance = False, fig=True, roc = False, draw_hist = False, add_synthetic = False)
	# plt.ion()

	# pdb.set_trace()

	# run_all(multi_position = False, use_tex_backend = use_tex_backend, noise_from_class = 1, average_model= False, show_training_variance = False, fig = False)
	# plot_fig()

if __name__ == '__main__':
	main()
	
