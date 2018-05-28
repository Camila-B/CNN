import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import torch.nn.init
from torch.nn.init import calculate_gain
from torchvision import datasets, models, transforms

import random
import pickle

import time
import os
from PIL import Image 
from PIL import ImageFilter


from sklearn.utils.extmath import cartesian

from utilities import linear_ini
from utilities import adjust_lr
from utilities import N_prediction
from utilities import N_prediction_2
from utilities import L2_weights_norm
from utilities import set_parameters

cuda = True

########################################################################################################################
########################################################################################################################
########################################################################################################################

def text_plot(ax_, string, alpha, color, fontsize):
	delta = [(ax_.get_xlim()[1] - ax_.get_xlim()[0]), (ax_.get_ylim()[1] - ax_.get_ylim()[0])]
	xt = alpha[0] * delta[0] + ax_.get_xlim()[0]
	yt = alpha[1] * delta[1] + ax_.get_ylim()[0]    
	ax_.text(xt, yt, string, fontsize=fontsize,color=color,clip_on=True)

def batch_prediction(data_batch,model,criterion): 
	loss = 0
	correct = 0
	total = 0
	for batch_idx, (inputs, targets) in enumerate(data_batch):
		targets = targets.squeeze()
		if cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = model(inputs)
		loss += criterion(outputs, targets).data[0]
		total += targets.size(0)

		_, pred = torch.max(outputs.data, 1)
		pred = Variable(pred)
		correct += float((pred == targets).sum())

	return loss/total, 100*correct/total, total - correct

def parameters(i,p,k,s):
	return (i + 2*p - k)/s + 1

def standardized(x, mean, std):
	'''
	mean and std dimension : 1 x 3 x 1 x 1 
	see function mean_std
	'''
	return torch.div((x - mean), std)

def mean_std(x):
	'''
	Returns the mean and std of all pixels, 1 x 3 x 1 x 1
	'''
	n = x.shape[0]
	x = x.permute(1,0,2,3).contiguous()
	mean = torch.mean(x.view(3,n*64*64),1)[None,:,None,None]
	std = torch.std(x.view(3,n*64*64),1)[None,:,None,None]
	return mean, std

def weights_init_1d(m):
	classname = m.__class__.__name__

	if isinstance(m, nn.BatchNorm1d):
		m.weight.data.normal_(0, 1)
		m.bias.data.fill_(0)		

	elif isinstance(m, nn.Linear):
		nn.init.xavier_uniform(m.weight.data, gain=calculate_gain('relu'))
		nn.init.constant(m.bias.data, 0)

def weights_init_2d(m):
	classname = m.__class__.__name__

	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_uniform(m.weight.data, mode='fan_in')
		nn.init.constant(m.bias.data, 0)
	elif isinstance(m, nn.BatchNorm2d):
		m.weight.data.normal_(0, 1)
		m.bias.data.fill_(0)

def save_model(m, optim, e_, ckpt_fname):
		state_dict = m.state_dict()
		for key in state_dict.keys():
			state_dict[key] = state_dict[key].cpu()
        
		torch.save({
			'epoch': e_,
			'state_dict': state_dict},
			ckpt_fname)
			# 'optimizer': optim

class MultipleOptimizer(object):
	def __init__(self, *op):
		self.optimizers = op

	def zero_grad(self):
		for op in self.optimizers:
			op.zero_grad()

	def step(self):
		for op in self.optimizers:
			op.step()

def opt_func(opt):
	if opt == 'adam':
		optimizer = MultipleOptimizer(
			optim.Adam(model.features.parameters(), lr=0.001), 
			optim.Adam(model.classifier.parameters(), lr=0.01))
	elif opt == 'sgd':
		optimizer = MultipleOptimizer(
			optim.SGD(model.features.parameters(), lr=0.01, momentum=0.9), 
			optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9))
	return optimizer

def train_model(nb_epochs, directory, filename):
	
	acc_optimal = 0
	loss_train = np.empty([nb_epochs])
	loss_valid = np.empty([nb_epochs])
	error_train = np.empty((nb_epochs)) 
	error_valid = np.empty((nb_epochs)) 
	acc_train = np.empty((nb_epochs)) 
	acc_valid = np.empty((nb_epochs)) 
	
	for epoch in range(nb_epochs):
		model.train()
		for batch_idx, (inputs, targets) in enumerate(train_batch):
			targets = targets.squeeze()
			inputs, targets = Variable(inputs), Variable(targets)
			
			if cuda:
				inputs, targets = inputs.cuda(), targets.cuda()
	
			optimizer.zero_grad()
			outputs = model(inputs)
	
			loss = criterion(outputs, targets)
			loss.backward()
			optimizer.step()
	
		model.eval()
		loss_train[epoch], acc_train[epoch], error_train[epoch] = batch_prediction(train_batch, model, criterion)
		loss_valid[epoch], acc_valid[epoch], error_valid[epoch] = batch_prediction(valid_batch, model, criterion) 
		
		if (acc_valid[epoch] > acc_optimal):
			acc_optimal = acc_valid[epoch]
			epoch_optimal = epoch
			torch.save(model.state_dict(), directory+'/Model_optimal.pth')
			
		print('Epoch {}, loss = {},  {}, accuracy = {},  {}'.format(epoch, loss_train[epoch], loss_valid[epoch], acc_train[epoch], acc_valid[epoch]))
	print('done!')   

	with open(directory+'/'+filename, 'wb') as  f: 
		pickle.dump([loss_train, acc_train, error_train, loss_valid, acc_valid, error_valid, epoch_optimal, acc_optimal], f)

########################################################################################################################
########################################################################################################################
########################################################################################################################

transform = transforms.Compose([transforms.ToTensor()])

dd = torchvision.datasets.ImageFolder('datasets/test_64x64/',transform)
test_data = torch.Tensor(len(dd),3,64,64) 
test_labels = torch.Tensor(len(dd),1) 

for i, dd_ in enumerate(dd):
	test_data[i] = dd_[0]
	test_labels[i] = dd_[1]

test_data_mean, test_data_std = mean_std(test_data)
test_data = standardized(test_data, test_data_mean, test_data_std)
test_labels = test_labels.long()

dd = torchvision.datasets.ImageFolder('datasets/valid_64x64/',transform)

valid_data = torch.Tensor(len(dd),3,64,64) 
valid_labels = torch.Tensor(len(dd),1) 

for i, dd_ in enumerate(dd):
    valid_data[i] = dd_[0]
    valid_labels[i] = dd_[1]

valid_data_mean, valid_data_std = mean_std(valid_data)
valid_data = standardized(valid_data, valid_data_mean, valid_data_std)
valid_labels = valid_labels.long()

dd = torchvision.datasets.ImageFolder('datasets/train_64x64/',transform)

train_data = torch.Tensor(len(dd),3,64,64) 
train_labels = torch.Tensor(len(dd),1) 

for i, dd_ in enumerate(dd):
    train_data[i] = dd_[0]
    train_labels[i] = dd_[1]

train_data_mean, train_data_std = mean_std(train_data)
train_data = standardized(train_data, train_data_mean, train_data_std)
train_labels = train_labels.long()


batch_size = 64
train_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(train_data,train_labels), batch_size=batch_size, shuffle=True)
valid_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(valid_data,valid_labels), batch_size=batch_size, shuffle=False)
test_batch = torch.utils.data.DataLoader(data_utils.TensorDataset(test_data,test_labels), batch_size=batch_size, shuffle=False)    

print('Data loaded!')


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
#########################
#########################                            
#########################
#########################
#########################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################


nb_epochs = 100
nb_epochsv2 = 50

print('Model A')
print('kaiming initialization')
from Q2_model_a import Classifier

model = Classifier()
if cuda:
    model = model.cuda()
model.apply(weights_init_2d)
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_A_Adam'
filename = 'log_A'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochs, directory, filename)

# ##########################################################################################

print('Pretrained initialization')
from Q2_model_a import Classifier_pretrained

model = Classifier_pretrained()
if cuda:
    model = model.cuda()
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_A_Pretrained_Adam'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochsv2, directory, filename)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

print('Model B')
print('kaiming initialization')
from Q2_model_b import Classifier

model = Classifier()
if cuda:
    model = model.cuda()
model.apply(weights_init_2d)
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_B_Adam'
filename = 'log_B'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochs, directory, filename)

# ##########################################################################################

print('Pretrained initialization')
from Q2_model_b import Classifier_pretrained

model = Classifier_pretrained()
if cuda:
    model = model.cuda()
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = MultipleOptimizer(
	optim.Adam(model.features.parameters(), lr=0.001), 
	optim.Adam(model.classifier.parameters(), lr=0.01))

directory = 'SaveData/Q2/Model_B_Pretrained_Adam'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochsv2, directory, filename)

# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################

print('Model D')
print('kaiming initialization')
from Q2_model_d import Classifier

model = Classifier()
if cuda:
    model = model.cuda()
model.apply(weights_init_2d)
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_D_Adam'
filename = 'log_D'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochs, directory, filename)

# ##########################################################################################

print('Pretrained initialization')
from Q2_model_d import Classifier_pretrained

model = Classifier_pretrained()
if cuda:
    model = model.cuda()
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_D_Pretrained_Adam'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochsv2, directory, filename)

# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################
# ##########################################################################################

print('Model E')
print('kaiming initialization')
from Q2_model_e import Classifier

model = Classifier()
if cuda:
    model = model.cuda()
model.apply(weights_init_2d)
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_E_Adam'
filename = 'log_E'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochs, directory, filename)

# ##########################################################################################

print('Pretrained initialization')
from Q2_model_e import Classifier_pretrained

model = Classifier_pretrained()
if cuda:
    model = model.cuda()
model.apply(weights_init_1d)
criterion = nn.CrossEntropyLoss()
optimizer = opt_func('adam') 

directory = 'SaveData/Q2/Model_E_Pretrained_Adam'
if not os.path.exists(directory):
	os.makedirs(directory)

train_model(nb_epochsv2, directory, filename)








