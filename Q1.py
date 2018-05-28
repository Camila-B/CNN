import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms
from torch.nn.init import calculate_gain

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import random
import pickle
import gzip

import time
import datetime

from sklearn.utils.extmath import cartesian

from utilities import linear_ini
from utilities import adjust_lr
from utilities import prediction
from utilities import prediction_conv
from utilities import N_prediction
from utilities import N_prediction_2
from utilities import L2_weights_norm
from utilities import set_parameters
from utilities import weights_init_1d
from utilities import weights_init_2d

########################################################################################################################
########################################################################################################################
########################################################################################################################


import pickle

with open('datasets/mnist.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, valid, test = u.load()

train_x, train_y = train
valid_x, valid_y = valid
test_x, test_y = test

train_x = train_x.reshape(50000,28,28)
valid_x = valid_x.reshape(10000,28,28)
test_x = test_x.reshape(10000,28,28)

train_data = torch.FloatTensor(train_x)
train_data_ = Variable(train_data.view(-1,784))
train_labels = torch.FloatTensor(train_y).long()
train_labels_ = Variable(train_labels.view(-1))

valid_data = torch.FloatTensor(valid_x)
valid_data_ = Variable(valid_data.view(-1,784))
valid_labels = torch.FloatTensor(valid_y).long()
valid_labels_ = Variable(valid_labels.view(-1))

test_data = torch.FloatTensor(test_x)
test_data_ = Variable(valid_data.view(-1,784))
test_labels = torch.FloatTensor(test_y).long()
test_labels_ = Variable(valid_labels.view(-1))

########################################################################################################################
########################################################################################################################
########################################################################################################################

class MLPLinear(nn.Module):
	def __init__(self, dimensions, dp, cuda):
		super(MLPLinear, self).__init__()
		self.h0 = int(dimensions[0])
		self.h1 = int(dimensions[1])
		self.h2 = int(dimensions[2])       

		self.fc1 = torch.nn.Linear(self.h0, self.h1)
		self.fc2 = torch.nn.Linear(self.h1, self.h2)
		self.fc2_drop = nn.Dropout(p=dp)
		self.relu = nn.ReLU()
		self.criterion = nn.CrossEntropyLoss()
		self.cuda = cuda

		if cuda: 
			self.fc1.cuda()
			self.fc2.cuda()
			self.fc2_drop.cuda()
			self.relu.cuda()
			self.criterion.cuda()

	def initialization(self,method):
		self.fc1 = linear_ini(self.fc1,method)
		self.fc2 = linear_ini(self.fc2,method)

	def input_shape(self,x,y):
		x = (x.view(-1,784))
		y = (y)  
		if self.cuda : 
			x = Variable(x.cuda())
			y = Variable(y.cuda())
		else: 
			x = Variable(x)
			y = Variable(y)
		return x,y

	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.fc2_drop(out)
		return  out

########################################################################################################################
########################################################################################################################
########################################################################################################################


class Convolution(nn.Module):
	def __init__(self):
		super(Convolution, self).__init__()
		
		self.criterion = nn.CrossEntropyLoss()
		self.conv = nn.Sequential(
		nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1),
		nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True),
		nn.Dropout(p=0.5),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=(2, 2), stride=2),
	            
		nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1),
		nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True),
		nn.Dropout(p=0.5),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=(2, 2), stride=2),
	            
		nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
		nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
		nn.Dropout(p=0.5),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=(2, 2), stride=2),
	            
		nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
		nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
		nn.Dropout(p=0.5),
		nn.ReLU(),
		nn.MaxPool2d(kernel_size=(2, 2), stride=2)
		)

		self.fc1 = nn.Linear(128, 10)

	def input_shape(self,x,y):
		return x, y

	def initialization(self,method):
		self.fc1 = linear_ini(self.fc1,method)

	def forward(self, x):
		return self.fc1(self.conv(x).squeeze())
		

########################################################################################################################
########################################################################################################################
########################################################################################################################

def q1_a():
	print('Compiling q1 (a)')
	cuda = True
	lr0 = 0.02
	batch_size = 64
	nb_epochs = 100
	
	weight_decay_0 = 2.5 * batch_size / train_data.shape[0]
	
	
	model = MLPLinear([784, 800, 10], 0, cuda) 
	model.initialization('glorot')
	# optimizer = optim.SGD(model.parameters(), lr=lr0, weight_decay=weight_decay_0)
	optimizer = optim.SGD(model.parameters(), lr=lr0)
	
	train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=batch_size, shuffle=True)
	valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
	test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)

	error_train = np.empty((nb_epochs)) 
	error_valid = np.empty((nb_epochs)) 
	acc_train = np.empty((nb_epochs)) 
	acc_valid = np.empty((nb_epochs)) 
	L2norm = np.empty((nb_epochs))
		
	for e_ in range(nb_epochs):
		for batch_idx, (x,y) in enumerate(train_batch):
			xt,yt = model.input_shape(x,y)
			pred_batch = model.forward(xt)          
			optimizer.zero_grad()
			loss_batch = model.criterion(pred_batch, yt)
			loss_batch.backward()
			optimizer.step()
				
		acc_train[e_], error_train[e_] = prediction(train_batch,model)
		acc_valid[e_], error_valid[e_] = prediction(valid_batch,model)
		L2norm[e_] = L2_weights_norm(model)
		
		if e_%10==0: print('Epoch #%.f, acc_train = %.5f, error_train = %.f'%(e_,acc_train[e_], error_train[e_]))
	
	acc_test, error_test = prediction(test_batch,model)

	print('q1(a), code compiled')

	return acc_train, acc_valid, error_train, error_valid, L2norm, acc_test, error_test


########################################################################################################################
########################################################################################################################

def q1_b_i(): 
	print('Compiling q1 (b) (i)')
	
	cuda = True
	lr0 = 0.02
	batch_size = 64
	nb_epochs = 100
	
	weight_decay_0 = 2.5 * batch_size / train_data.shape[0]
	
	model = MLPLinear([784, 800, 10], 0, cuda) 
	model.initialization('glorot')
	optimizer = optim.SGD(model.parameters(), lr=lr0, weight_decay=weight_decay_0)
	
	train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=batch_size, shuffle=True)
	valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
	test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)

	error_train = np.empty((nb_epochs)) 
	error_valid = np.empty((nb_epochs)) 
	acc_train = np.empty((nb_epochs)) 
	acc_valid = np.empty((nb_epochs)) 
	L2norm = np.empty((nb_epochs))
		
	for e_ in range(nb_epochs):
		for batch_idx, (x,y) in enumerate(train_batch):
			xt,yt = model.input_shape(x,y)
			pred_batch = model.forward(xt)          
			optimizer.zero_grad()
			loss_batch = model.criterion(pred_batch, yt)
			loss_batch.backward()
			optimizer.step()
				
		acc_train[e_], error_train[e_] = prediction(train_batch,model)
		acc_valid[e_], error_valid[e_] = prediction(valid_batch,model)
		L2norm[e_] = L2_weights_norm(model)
		
		if e_%10==0: print('Epoch #%.f, acc_train = %.5f, error_train = %.f'%(e_,acc_train[e_], error_train[e_]))
	
	torch.save(model.state_dict(), 'SaveData/Q1/model_Q1b_i.pth')

	set_parameters(model,'fc2.weight',0.5)

	acc_test_i, error_test_i = prediction(test_batch,model)
	
	print('Done!')
	return acc_test_i, error_test_i


def q1_b_ii(): 
	print('Compiling q1 (b) (ii)')
	
	cuda = True
	lr0 = 0.02
	batch_size = 64
	nb_epochs = 100
	
	weight_decay_0 = 2.5 * batch_size / train_data.shape[0]
	
	model = MLPLinear([784, 800, 10], 0.5, cuda) 
	model.initialization('glorot')
	optimizer = optim.SGD(model.parameters(), lr=lr0, weight_decay=weight_decay_0)
	
	train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=batch_size, shuffle=True)
	valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
	test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)

	error_train = np.empty((nb_epochs)) 
	error_valid = np.empty((nb_epochs)) 
	acc_train = np.empty((nb_epochs)) 
	acc_valid = np.empty((nb_epochs)) 
	L2norm = np.empty((nb_epochs))
		
	for e_ in range(nb_epochs):
		for batch_idx, (x,y) in enumerate(train_batch):
			xt,yt = model.input_shape(x,y)
			pred_batch = model.forward(xt)          
			optimizer.zero_grad()
			loss_batch = model.criterion(pred_batch, yt)
			loss_batch.backward()
			optimizer.step()
				
		acc_train[e_], error_train[e_] = prediction(train_batch,model)
		acc_valid[e_], error_valid[e_] = prediction(valid_batch,model)
		L2norm[e_] = L2_weights_norm(model)
		
		if e_%10==0: print('Epoch #%.f, acc_train = %.5f, error_train = %.f'%(e_,acc_train[e_], error_train[e_]))
	
	torch.save(model.state_dict(), 'SaveData/Q1/model_Q1b_ii.pth')

	set_parameters(model,'fc2.weight',0.5)
	acc_test_i, error_test_i = prediction(test_batch,model)
	

	# (ii) and (iii), see N_prediction and N_prediction_2 in utilities.py. 
	N = [1,2,3,4,5,8,10,15,20,30,40,50,60,70,80,90,100]
	acc_test_N = np.array([])
	error_test_N = np.array([])
	acc_test_N_2 = np.array([])
	error_test_N_2 = np.array([])

	for n in N: 
		acc,err,acc2,err2 = N_prediction(test_batch,model,n)
		acc_test_N = np.append(acc_test_N,acc)
		error_test_N = np.append(error_test_N,err)

		# acc_test_N_2 = np.append(acc_test_N_2,acc2)
		# error_test_N_2 = np.append(error_test_N_2,err2)

		acc,err = N_prediction_2(test_batch,model,n)
		acc_test_N_2 = np.append(acc_test_N_2,acc)
		error_test_N_2 = np.append(error_test_N_2,err)

	print('Done!')
	return acc_test_i, error_test_i, acc_test_N, error_test_N, acc_test_N_2, error_test_N_2


########################################################################################################################
########################################################################################################################

def q1_c(): 
	print('Compiling q1 (c)')
	
	cuda = True
	lr0 = 0.02
	batch_size = 64
	nb_epochs = 10
	
	weight_decay_0 = 2.5 * batch_size / train_data.shape[0]
	
	model = Convolution()
	if cuda: 
		model.cuda()

	model.apply(weights_init_2d)
	model.apply(weights_init_1d)

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	
	train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=batch_size, shuffle=True)
	valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
	test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)

	error_train = np.empty((nb_epochs)) 
	error_valid = np.empty((nb_epochs)) 
	acc_train = np.empty((nb_epochs)) 
	acc_valid = np.empty((nb_epochs)) 
	L2norm = np.empty((nb_epochs))
		
	for e_ in range(nb_epochs):
		for batch_idx, (x,y) in enumerate(train_batch):
			x = Variable(x[:, None, :, :])
			y = Variable(y)
			if cuda:
				x, y = x.cuda(), y.cuda()
			pred_batch = model.forward(x)          
			optimizer.zero_grad()
			loss_batch = model.criterion(pred_batch, y)
			loss_batch.backward()
			optimizer.step()
		
		acc_train[e_], error_train[e_] = prediction_conv(train_batch, model, cuda)
		acc_valid[e_], error_valid[e_] = prediction_conv(valid_batch, model, cuda)
		L2norm[e_] = L2_weights_norm(model)
		
		if e_%1==0: print('Epoch #%.f, acc_train = %.5f, error_train = %.f'%(e_,acc_train[e_], error_train[e_]))
	
	torch.save(model.state_dict(), 'SaveData/Q1/model_Q1c.pth')

	acc_test, error_test = prediction_conv(test_batch, model, cuda)
	
	print('Done!')
	return acc_train, acc_valid, error_train, error_valid, L2norm, acc_test, error_test


########################################################################################################################
########################################################################################################################


if __name__ == "__main__":
	pass

	# acc_train, acc_valid, error_train, error_valid, L2norm, acc_test, error_test = q1_c()

	# filename = 'SaveData/Q1/MNIST_Error_Accuracy_Convolution'
	# with open(filename, 'wb') as  f: 
	# 	pickle.dump([acc_train, acc_valid, error_train, error_valid, L2norm, acc_test, error_test], f)
	
	
	# print('Q1a')
	# acc_train, acc_valid, error_train, error_valid, L2norm, acc_test, error_test = q1_a()
	# filename = 'SaveData/Q1/MNIST_Error_Accuracy_L2norm_No_Regularization_i'
	# with open(filename, 'wb') as  f: 
	# 	pickle.dump([acc_train, acc_valid, error_train, error_valid, L2norm, acc_test, error_test], f)
	
	
	
	####################################################################################
	####################################################################################
	####################################################################################
	
	