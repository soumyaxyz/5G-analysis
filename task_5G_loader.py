import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# from KitNET.KitNET import KitNET
# from KitNET.distNET import distNET
from KitNET.multiNET import multiNET
NET = multiNET
import pdb, traceback, csv
import math, re
import copy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

import matplotlib, importlib
from matplotlib import pyplot as plt, colors

from math import dist
from sklearn.preprocessing import MinMaxScaler


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

class Task1_data_loader():
	"""docstring for Task1_data_loader"""
	def __init__(self, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, 
				shuffle = False, noise_from_class = 1, mask=[0],#, 3, 9, 10,  14, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,   4,5,8],#[0,*range(7,32)],
				 add_synthetic=True, guard=50 ):  #,*range(7,32) #,7,8,18,17,*range(20,32)
		self.poison_persent 	= poison_persent
		self.training_fraction 	= training_fraction
		assert placement_percent >=0 and placement_percent <=1, 'placement_percent should be between 0(leftmost) and 1(rightmost)'
		self.placement_percent 	= placement_percent
		self.shuffle 			= shuffle
		data_subfolder = '5G/training_data'
		self.data_files = [	#data_subfolder+"/SNR_data_10K__10feet_0degree.csv",  	# normal 
							data_subfolder+"/SNR_data_5000_10feet_0degree.csv",  	# normal 
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
		# data_subfolder = '5G'
		# self.data_files = [	data_subfolder+"/SNR_data_10000_10feet.csv",  	# normal 
		# 					# data_subfolder+"/SNR_data_5000__10feet_45degree.csv",
		# 					# data_subfolder+"/SNR_data_5000__10feet_90degree.csv",
		# 					# data_subfolder+"/SNR_data_5000__10feet_180degree.csv",
		# 					data_subfolder+"/SNR_data_10000_20feet.csv", 		
		# 					# data_subfolder+"/SNR_data_5000__20feet_45degree.csv",
		# 					# data_subfolder+"/SNR_data_5000__20feet_90degree.csv",	
		# 					data_subfolder+"/SNR_data_10000_30feet.csv", 		
		# 					data_subfolder+"/SNR_data_10000_40feet.csv", 		
		# 					data_subfolder+"/SNR_data_10000_50feet.csv"
		# 					]

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
								'Announce frames sent', 'BI announce acked', 'BI announce not acked', 'MPDUs received	', 
								'MPDUs transmitted ', 'Inactive time', 'BH2 temperature'
							]
		self.feature_names = np.delete(self.feature_names,mask)
		
		
	def loadfile(self, filename, mask = []):
		df  = pd.read_csv(filename, header = None, skiprows = self.guard)
		data = df.values

		inc = [17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29]
		rows, cols = data.shape
		import copy
		data_copy = copy.deepcopy(data)
		for r in range(rows):
			for c in range(cols):
				if c in inc:
					data[r,c]-=data_copy[r-1,c]

		# pdb.set_trace()
		data = np.delete(data,mask,1)
		tokens =  re.findall(r'\d+', filename)
		try:
			label = int(tokens[3])
		except IndexError as e:
			label = 0
		label = [int(tokens[2])*-1, label]  # denoting the distance as negative and degrees as posative
		# label = [int(tokens[2]+tokens[3]), -1]
		# print(label)
		return data, label

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


	def loadSupervisedData(self):

		data_files = self.data_files
		data_count = len(data_files)

		training_size_equalized = np.inf
		total_training_size = 0

		

		for i in range(data_count):
			class_data, _ = self.loadfile(data_files[i], self.mask)
			training_size 	= round(class_data.shape[0]*self.training_fraction)
			total_training_size+= training_size
			if training_size_equalized> training_size:
				training_size_equalized = training_size


		holdout = round(training_size_equalized/10) ###################### temporary

		training_size = training_size_equalized

		self.unique_class = set()

		# pdb.set_trace()
		self.class_end_index_train=dict()
		self.class_end_index_test=dict()
		
		for i in range(data_count):
			# if i == 0:
			# 	training_size = holdout+training_size_equalized
			# 	if training_size>class_data.shape[0]:
			# 	  training_size =holdout+round(class_data.shape[0]*.1)
			# else:
			# 	training_size = training_size_equalized




			class_data, class_label = self.loadfile(data_files[i], self.mask)
			for l in class_label:
				# print(l, class_label)
				self.unique_class.add(l)

			# training_size 	= round(class_data.shape[0]*self.training_fraction)
			# print(f'{training_size=},{class_data.shape[0]=}')
			try:
				if i == 0 :
					pretrain_data		= class_data[:holdout]
					pretrain_labels 	= [class_label]*(holdout)


					train_data 			= class_data[holdout:training_size]
					train_labels 		= [class_label]*(training_size-holdout)

					test_data 			= class_data[training_size:]
					test_labels 		= [class_label]*(class_data.shape[0]-training_size)  
									
				else:
					pretrain_data		= np.vstack((pretrain_data, class_data[:holdout]))
					pretrain_labels 	= np.vstack((pretrain_labels,[class_label]*(holdout)))

					train_data 			= np.vstack((train_data, class_data[holdout:training_size]))				
					train_labels		= np.vstack((train_labels, [class_label]*(training_size-holdout)))

					test_data 			= np.vstack((test_data, class_data[training_size:]))				
					test_labels			= np.vstack((test_labels, [class_label]*(class_data.shape[0]-training_size)))

				class_name =  str(class_label[0]*-1)+'ft '+str(class_label[1])+'°'

				self.class_end_index_train[class_name]=len(train_labels)
				self.class_end_index_test[class_name]=len(test_labels)

				# print(f'{str(class_name)} class end_index_train {len(train_labels)}')
			except Exception as e:
				traceback.print_exc()
				pdb.set_trace()

		# pdb.set_trace()
		self.class_end_index_train['start']=0

		self.class_end_index_train = { k:len(pretrain_labels)+v for k,v in self.class_end_index_train.items()}


		train_data 			= np.vstack((pretrain_data,train_data))				
		train_labels		= np.vstack((pretrain_labels,train_labels))


		# pdb.set_trace()
		self.unique_class =	sorted(self.unique_class)
		self.training_size = train_labels.shape[0]
		self.training_size_initial = self.training_size	
		
		if self.poison_persent>0:			
			# data, labels 	= self.poison(data, labels)
			raise NotImplementedError

		
		if self.shuffle:
			# data, labels 	= self.shuffle_training_data_only(data, labels)
			raise NotImplementedError


		scaler = MinMaxScaler()
		
		scaler.fit(train_data)

		train_data = scaler.transform(train_data)
		test_data = scaler.transform(test_data)

		if self.add_synthetic:
			# normalized_data, labels = self.add_synthetic_training_data(normalized_data, labels)
			raise NotImplementedError




		train_data 		= train_data.astype(np.float64)	
		test_data 		= test_data.astype(np.float64)

		# pca = PCA(n_components=10)
		# train_data = pca.fit_transform(train_data)
		# test_data = pca.transform(test_data)
		pdb.set_trace()
		


		return train_data, test_data, train_labels, test_labels

class Task1():
	"""docstring for Task1"""
	def __init__(self, poison_persent = 0, training_fraction = 0.6, placement_percent = 0, shuffle = False, 
				noise_from_class = 1, average_model= True, window_size= 500, show_training_variance = False, draw_hist = False, add_synthetic = False, guard = 0, scale_output=True):
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
		self.guard 				= guard
		self.scale_output		= scale_output

	def loadData(self):

		data_loader =  	Task1_data_loader(self.poison_persent, self.training_fraction, self.placement_percent, self.shuffle, self.adversarial_class, add_synthetic = self.add_synthetic, guard = self.guard)
		# data, labels  		= data_loader.loadData()
		train_data, test_data, train_labels, test_labels  		= data_loader.loadSupervisedData()
		data 			= np.vstack((train_data,test_data))
		labels 			= np.vstack((train_labels,  test_labels))

		self.data_files 	= data_loader.data_files
		self.filename		= data_loader.filename
		self.feature_names 	= data_loader.feature_names


		self.classNames       = data_loader.unique_class
		self.class_end_index_train = data_loader.class_end_index_train
		self.class_end_index_test = data_loader.class_end_index_test

		# for filename in data_loader.filename:
		# 	tokens =  re.findall(r'\d+', filename)



		# pdb.set_trace()
		self.training_size 	= data_loader.training_size#*3	
		self.training_size_initial  = data_loader.training_size_initial
		self.data 		= data
		self.labels 	= labels

		data_shape 				= self.data.shape
		self.num_samples 		= data_shape[0]
		self.num_features 		= data_shape[1]

		# pdb.set_trace()

		# results = tanRMSEs[training_size+1:]
		# gold 	= labels[training_size+1:]

		return self.data, self.labels


	def train_and_predict(self):

		# pdb.set_trace()
		print(f'{self.training_size=}')
		
		max_autoencoder_size	= 10
		FM_grace_period			= round(self.training_size_initial/10)
		AD_grace_period			= self.training_size - FM_grace_period
		learning_rate			= 0.1
		hidden_ratio			= 0.75
		corruption_level 		= 0.2

		AnomDetector = NET(self.num_features, self.classNames, max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate, corruption_level)
		self.AD = AnomDetector
		# pdb.set_trace()
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
		# fixed_RMSEs  = []
		list_of_RMSEs = []
		# pbar = tqdm(total=num_samples, leave=True)
		self.predictions = []

		# distance_classes_idx = [i for i, n in enumerate(self.classNames) if n<0]
		# angle_classes_idx = [i for i, n in enumerate(self.classNames) if n>=0]
		distance_classes_until = self.classNames.index(0)
		

		for i in tqdm(range(self.num_samples), leave=False):
			# data_shape
			
			try:
				# process KitNET
				rmse, msg = AnomDetector.process(self.data[i],self.labels[i])  # will train during the grace periods, then execute on all the rest.


				
				# if i> self.training_size:
				# 	rmse *=-1

				# if msgLast!= msg:
				# 	msgLast = msg
				# 	pbar.set_description(msg)

				if isinstance(rmse,int):
					RMSEs.append(rmse)
					self.predictions.append(self.labels[i]) # Training data, prediction irrelavant
				elif isinstance(rmse,list) :
					if len(rmse)<=2:
						RMSEs.append(rmse[0])
						self.predictions.append(self.labels[i]) # Training data, prediction irrelavant
					else:
						dist_idx = np.argmin(rmse[:distance_classes_until])
						dist_class = self.classNames[dist_idx]
						angle_idx = np.argmin(rmse[distance_classes_until:])
						angle_class = self.classNames[distance_classes_until+angle_idx]

						try:
							assert angle_class>=0, 'angle error'
							assert dist_class<=0, 'distance_error'
						except Exception as e:
							traceback.print_exc()
							pdb.set_trace()

						self.predictions.append([dist_class, angle_class])
						# pdb.set_trace()
						RMSEs.append( rmse[dist_idx])
						if not list_of_RMSEs:
							for j in range(len(rmse)):
								list_of_RMSEs.append(copy.deepcopy(RMSEs))
							# pdb.set_trace()
						else:
							for j in range(len(rmse)):
								list_of_RMSEs[j].append(rmse[j])
						assert len(list_of_RMSEs[j])==len(RMSEs)
					# pdb.set_trace()


				


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

				
				if self.average_model:
					raise NotImplementedError

				# if self.average_model and i>self.training_size:
				# 	actual_rmse = rmse 
				# 	rmse = np.mean(np.tan( RMSEs[max(i-window_size+1,0):i+1]) )
				# 	# if i%100 ==0:
				# 	# 	print(i, actual_rmse, '>', i-window_size+1, ':',i+1, rmse)
					
				# fixed_RMSEs.append(rmse)
				assert i+1 == len(RMSEs)
			except Exception as e:
				traceback.print_exc()
				pdb.set_trace()

			# if i%100==0:
				# pbar.set_description(msg)
				# pbar.update(100)
				# pdb.set_trace()
			# if i>900:
				# pdb.set_trace()
		# pdb.set_trace()
		self.clusters = AnomDetector.v
		if not list_of_RMSEs:
			list_of_RMSEs.append(copy.deepcopy(RMSEs))

		assert  AnomDetector.classNames == self.classNames, 'class name mismatch'

		if self.training_variance:
			window = std_window_size*2
			self.training_consistancy_std =  np.mean(self.STD[-window:])/np.mean(self.STD[window:]) 
			self.training_consistancy_mean =  np.mean(self.Mean[-window:])/np.mean(self.Mean[window:])
			
			print(f'Training std consistancy :{self.training_consistancy_std:.4f}')
			print(f'Training mean consistancy :{self.training_consistancy_mean:.4f}')


		# tanRMSEs =  np.tanh(fixed_RMSEs)

		# results = tanRMSEs[training_size+1:]
		# gold 	= self.labels[training_size+1:]
		self.results_raw 	= np.tanh(RMSEs)
		for i in range(len(list_of_RMSEs)):
			list_of_RMSEs[i] = np.tanh(list_of_RMSEs[i])
		# 	print(list_of_RMSEs[i][1555])
		# pdb.set_trace()
		self.predictions = np.array(self.predictions)
		self.results 		= list_of_RMSEs
		# self.gold 			= np.round(np.tanh(self.labels))
		self.gold 			= np.array(self.labels)

		# self.gold 			= self.gold.T

		# pdb.set_trace()
		# results =  tanRMSEs
		# RMSEData = np.vstack((RMSEs,tanRMSEs))
		# df = pd.DataFrame(RMSEData)

		return self.gold, self.results

	def print_results(self, threshold = None):


		# distance_classes_idx = [i for i, n in enumerate(self.classNames) if n<0]
		# angle_classes_idx = [i for i, n in enumerate(self.classNames) if n>=0]

		# distance_results = self.results[distance_classes_idx]
		# angle_results = self.results[angle_classes_idx]
		# for i in distance_classes_idx:



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

		try:
			
			# pred = []
			# for result in self.results_raw:
			# 	if result>self.threshold:
			# 		pred.append(1)
			# 	else:
			# 		pred.append(0)

			
			# acc = accuracy_score(self.gold[self.training_size:], pred[self.training_size:])

			distance_acc = accuracy_score(self.gold[self.training_size:,0], self.predictions[self.training_size:,0])

			angle_acc = accuracy_score(self.gold[self.training_size:,1], self.predictions[self.training_size:,1])

			

			
			distance_classes_until = self.classNames.index(0)

			if len(self.classNames[:distance_classes_until])>1:
				print(f'distance accuracy score :{distance_acc:.3f}')

				cm = confusion_matrix(self.gold[self.training_size:,0], self.predictions[self.training_size:,0], labels=self.classNames[:distance_classes_until])
				plot_CM(cm, [x*-1 for x in self.classNames[:distance_classes_until]])

			if len(self.classNames[distance_classes_until:])>1:
				print(f'angle accuracy score :{angle_acc:.3f}')

				cm = confusion_matrix(self.gold[self.training_size:,1], self.predictions[self.training_size:,1], labels=self.classNames[distance_classes_until:])			
				plot_CM(cm, self.classNames[distance_classes_until:])
			


			# idx = []
			# start_at = self.training_size+1
			# idx.append([0,start_at])		
			# num_classes = int(np.max(self.labels))+1
			# for i in range(num_classes-1):
			# 	end_at = np.where(self.labels[self.training_size+1:]==i+1)[0][0]+self.training_size+1
			# 	idx.append([start_at,end_at])
			# 	start_at = end_at
			# idx.append([start_at,-1])

			# pdb.set_trace()

			# print('class wise accuracy:')
			# for i in range(1,len(idx)):
			# 	acc_i = accuracy_score(self.labels[idx[i][0]:idx[i][1]], self.predictions[idx[i][0]:idx[i][1]])
			# 	# rec_i = recall_score(self.gold[idx[i][0]:idx[i][1]], pred[idx[i][0]:idx[i][1]])
			# 	# pre_i = precision_score(self.gold[idx[i][0]:idx[i][1]], pred[idx[i][0]:idx[i][1]])
			# 	if math.isnan(acc_i):
			# 		pdb.set_trace()

			# 	print(f'\t{self.data_files[i-1][18:-4]}  :\t\t{acc_i:.3f}')#\t{pre_i:.3f}\t{rec_i:.3f}')

			# cm = confusion_matrix(self.labels[self.training_size:], self.predictions[self.training_size:]) #,[a[18:-4] for a in self.data_files])
			# print(cm)
		except Exception as e:
			traceback.print_exc()
			pdb.set_trace()

	def plot_results(self, threshold = None, fig = True): #threshold set by inspection
		if not threshold:			
			self.threshold = np.mean(self.Mean)			
		else:
			self.threshold = threshold	

		if self.classNames.index(0)>1:
			flag = 0
		else:
			flag = 1
		
		
		colors_set = ['green','red','blue','purple', 'yellow','cyan','brown','orange','blue','purple', 'yellow','cyan','brown','orange']
		if  len(self.data_files)>len(colors_set):		
			colors_set = [	'green','lime','springgreen','darkgreen',
							 'blue','deepskyblue','cyan','dodgerblue',
							 'violet','purple', 'magenta','darkviolet',
							 'brown', 'firebrick','lightcoral', 'red',
							 'orange', 'yellow', 'gold', 'khaki'
						]

		colors_set = colors_set[:len(self.data_files)]
		# classes = ['no_interference','nr_11dBm','lte_ul_11dBm','lte_dl_11dBm']
		# colors_set_ = colors_set[1:]+colors_set[:1]

		try:
			# plt.scatter(range(len(self.results_raw)),self.results_raw,s=1, marker='.',c=self.labels, cmap=colors.ListedColormap(colors_set) )#label='b')

			markers = ['.', 'o', '+', 'x', '*', '1', '2', '3', 'o', '+', 'x', '*', '1', '2', '3']
			# for k, result in enumerate(self.results):
			# 	# result = [r + k for r in result]
			# 	plt.scatter(range(len(result)),result,s=1, marker=markers[k],c=self.labels, cmap=colors.ListedColormap(colors_set) )#label='b')
				
			plt.scatter(range(self.num_samples), [0]*self.num_samples, s=.1, marker='.',c= self.gold[:,flag], cmap=colors.ListedColormap(colors_set) )
			# pdb.set_trace()
			# plt.scatter(range(self.num_samples), self.results_raw, s=.1, marker='.',c= self.gold[:,flag], cmap=colors.ListedColormap(colors_set) )

			# for i in range(self.numclass):
			# 	plt.scatter(range(self.num_samples), self.results[i], [i]*self.num_samples, s=.1, marker='.',c=self.labels, cmap=colors.ListedColormap(colors_set) )


			plt.scatter(range(self.num_samples), self.results_raw, s=.1, marker='.',c=self.predictions[:,flag], cmap=colors.ListedColormap(colors_set) )
			

			# plt.scatter(range(self.training_size,self.num_samples), self.results_raw[self.training_size:], s=.1, marker='.',c='k')


			# if self.average_model:
			# 	plt.scatter(range(self.training_size,self.num_samples), self.results[self.training_size:], s=.1, marker='.',c='k')


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
			plt.scatter([-1],[-1],s=3, marker='o', color=colors_set[i] , label=file[18:-4])
			# print(i,' ',file[7:],' ',colors_set[i])
		if self.average_model:
			plt.scatter([-1],[-1],s=3, marker='o', color='k' , label='running average')


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
		for i in range(num_classes-1):
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
			print(f'accuracy score for {file:s} :{acc:.3f}')
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

	def eval(self, threshold, fig, roc):
		self.loadData()
		self.train_and_predict()

		self.print_results(threshold)


		self.plot_results(threshold, fig)

		pdb.set_trace()
		# if self.draw_hist:
		# 	self.draw_histograms(500, False)
		# 	# self.draw_histograms(500)
		# 	# pdb.set_trace()

		roc_val = 0
		# roc_val = self.calc_roc()
		if roc:
			raise NotImplementedError
		# 	draw_roc_curve([[roc_val, 'ROC curve']] )


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

def run_all(multi_position = False, use_tex_backend = False, noise_from_class = 1, average_model= True, window_size = 500, show_training_variance = False, guard = 0):

	roc_Vals 	= []

	task = Task1(training_fraction = 0.6, shuffle=False, noise_from_class=noise_from_class, average_model=average_model, show_training_variance=show_training_variance, guard=guard)
	# task = Task1(poison_persent = 5, training_fraction = 0.6, placement_percent = .5, shuffle=False)
	roc_val, Tstd, Tmean = task.eval(threshold = .75, fig=False, roc=False)
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
				task = Task1(poison_persent = i, training_fraction = 0.6, placement_percent = j, shuffle=False, noise_from_class=noise_from_class, average_model=average_model, window_size= window_size, show_training_variance=show_training_variance, guard=guard)
				roc_val,  Tstd, Tmean= task.eval(threshold = None, fig=False, roc=False)

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

def pca(data, investigate):
	from sklearn.decomposition import PCA
	pca = PCA()
	pc =  pca.fit_transform(data)
	plt.figure()
	plt.plot(np.cumsum(pca.explained_variance_ratio_))
	plt.show()


	pca = PCA(n_components=6)
	new_data =  pca.fit_transform(data)

def explore_data(training_fraction = 0.6, placement_percent = 0.1, poison_persent = 5, noise_from_class = 1,  shuffle = False, 
	plot_training_classes = False, plot_stats= True, TSNE = False, use_gen =False):
	# matplotlib.use('TkAgg')

	try:		
		task1 = Task1(poison_persent, training_fraction , placement_percent, shuffle, noise_from_class)
		data, labels = task1.loadData()
		feature_names = task1.feature_names
		num_classes = len(task1.class_end_index_train)
		# num_classes = len( task1.classNames)#int(np.max(labels))+1

		# # pdb.set_trace()
		
		# train_idx = []
		# start_at = task1.training_size+1
		# for i in range( num_classes):
		# 	end_at = np.where(labels[:task1.training_size]==i)[0][0]
		# 	train_idx.append([start_at,end_at])
		# 	start_at = end_at

		# i +=1

		# test_idx = []
		# start_at = task1.training_size+1
		# for i in range( num_classes):
		# 	end_at = task1.training_size+np.where(labels[task1.training_size:]==i)[0][0]
		# 	test_idx.append([start_at,end_at])
		# 	start_at = end_at

		# i +=1


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
			name = task1.filename[0]
			# name = name[19:-4]
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
					name = task1.filename[i]
					# name = name[7:-4]
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
				colours = ['palegreen','green','navy','aqua','firebrick','cyan','darkred','darkturquoise','maroon','yellow','palegreen']
				# colours = ['green','green','navy','aqua','yellow','aqua','yellow','aqua','yellow','palegreen']

				from sklearn.manifold import TSNE		
				if use_gen:
					raise NotImplementedError
				print("Please wait, Computing TSNE ...")
				tsne = TSNE(n_components=2, n_iter=500)
				tsne_results = pd.DataFrame(tsne.fit_transform(data))
				# pdb.set_trace()

				# if use_gen:
				# 	plt.scatter(tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],0], tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],1],marker='.', c=colours[j+1], label='generated')

				for j in reversed(range(num_classes)):
					name = task1.filename[j]
					# name = name[18:-4]
					# print(hist_idx[j], hist_idx[j+1])
					# plt.scatter(tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],0], tsne_results.iloc[hist_idx[j][1]:hist_idx[j+1][1],1],marker='.', c=colours[j+1], label=name)
					plt.scatter(tsne_results.iloc[ np.where(labels==j+1)[0],0], tsne_results.iloc[ np.where(labels==j+1)[0],1],marker='.', c=colours[j+1], label=name)
				j=j+1
				# print(j)

				# plt.scatter(tsne_results.iloc[ np.where(labels==0),0], tsne_results.iloc[ np.where(labels==0),1],marker='o', c=colours[j+1], label=name)
				# plt.scatter(tsne_results.iloc[:hist_idx[0][1],0], tsne_results.iloc[:hist_idx[0][1],1], marker='.', c=colours[0], label='Train') #alpha=0.2


				plt.legend()
				plt.show()
				pdb.set_trace()
			else :
				if plot_training_classes:
					norm = data
				else:
					raise NotImplementedError
				# idx = [*range(norm.shape[0])]
				val = [12, 19, 11, 20, 6, 15, 2, 1, 7, 16, 13]#,5,4,8]

				train_idx = list(task1.class_end_index_train.values())
				train_class = list(task1.class_end_index_train.keys())
				# pdb.set_trace()


				idx = [*range(task1.class_end_index_train['start'],task1.training_size)]

				num_features =norm.shape[1]
				zeros= np.zeros(task1.training_size)

				j = 0
				for i in range(num_features):
					if i in val:
						sample = norm[task1.class_end_index_train['start']:task1.training_size,i]
						# sample = +3*i[float('nan') if x==0 else x for x in sample]
						sample = [float('nan') if x==0 else x+2*j for x in sample]
						plt.plot(idx, sample,'-')
						# plt.plot(idx, zeros+3*i,'.')
						j+=1


				

				if plot_training_classes:
					for j in range( num_classes-1):
						plt.plot([train_idx[j],train_idx[j]], [-1, 2*(i+1)],'--',color='k')
						plt.text(train_idx[j], 2*(i+1), train_class[j], ha='right',  va='center', rotation=0)#, fontsize=50)

				# for j in range( num_classes):
				# 	plt.plot([test_idx[j][1],test_idx[j][1]], [-1, 3*(i+1)],'k')
				# 	plt.text(test_idx[j][1], 3*(i+1), task1.filename[j], ha='left',  va='bottom', rotation=45)


				
				# print(f'{train_idx=},\n{test_idx=}')



				# plt.yticks([2*i for i in range(num_features)],feature_names, fontsize=50)
				# val = [12, 19, 11, 20, 6, 15, 2, 1, 7, 16, 13,5,4,8]
				for i in val:
					print(feature_names[i], i)

				mask = [0, 3, 9, 10,  14, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,   4,5,8]
				f_names = []
				for i in range(num_features):
					if i not in mask: 
					# 	f_names.append(feature_names[i])
					# else:
						pos = val.index(i)
						f_names.append(feature_names[i])

				# plt.yticks([2*i for i in range(num_features)],f_names)
				plt.yticks([2*i for i in range(len(val))],f_names)

				plt.show()
				pdb.set_trace()


				

	except Exception as e:
		traceback.print_exc()
		pdb.set_trace()
	# pdb.set_trace()

def run(poison_persent, training_fraction, placement_percent, shuffle=False, noise_from_class = 1, average_model= True, window_size= 500, 
	show_training_variance = False, fig=True, roc =True, draw_hist =False, add_synthetic = False, guard= 0, scale_output= False):

	roc_Vals 	= []

	# task = Task1(training_fraction = 0.6, shuffle=False)
	task = Task1(poison_persent, training_fraction , placement_percent, shuffle, noise_from_class, average_model, window_size, show_training_variance, draw_hist, add_synthetic, guard, scale_output)

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
	# explore_data(poison_persent = 0, training_fraction = 0.9, placement_percent = .5, plot_training_classes = True, plot_stats= False, TSNE = False, use_gen = False)
	run(poison_persent = 0, training_fraction = 0.8, placement_percent = .1, shuffle=False, noise_from_class = 1, 
		average_model= False, window_size= 100, show_training_variance = False, fig=True, roc = False, draw_hist = False, add_synthetic = False, guard = 0, scale_output= False)
	# plt.ion()

	# pdb.set_trace()

	# run_all(multi_position = False, use_tex_backend = use_tex_backend, noise_from_class = 1, average_model= False, show_training_variance = False)
	# plot_fig()

if __name__ == '__main__':
	main()
	
