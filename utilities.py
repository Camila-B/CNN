import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms
from torch.nn.init import calculate_gain

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


def set_parameters(model,param,factor):
    for name, p in model.named_parameters():
        if param in name: 
            p.data =  factor * p.data

def L2_weights_norm(model):
    weights = []
    for name, parameter in model.named_parameters():
        if 'weight' in name: 
            weights.append(parameter)

    return (torch.norm(weights[0]).data[0] + torch.norm(weights[1]).data[0])**2

def adjust_lr(optimizer,lr, epoch, total_epochs):
    lr = lr * (0.36 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def N_prediction(data_batch,model,N):
    softmax = nn.Softmax(dim = 1)
    correct = 0
    correct2 = 0
    total = 0
    
    for x,y in data_batch:
        out = 0
        out2 = 0
        x,y = model.input_shape(x,y)
        for i in range(N):
            out += model(x)
            out2 += softmax(out)
        out = softmax(out/N)
        out2 = out/N
        _, pred = torch.max(out.data, 1)
        _, pred2 = torch.max(out2.data, 1)
        pred = Variable(pred)
        pred2 = Variable(pred2)
        correct += float((pred == y).sum())
        correct2 += float((pred2 == y).sum())
        total += float(y.size(0))           

    return 100*correct/total, total - correct, 100*correct2/total, total - correct2

def N_prediction_2(data_batch,model,N):
    softmax = nn.Softmax(dim = 1)
    correct = 0
    total = 0
    for x,y in data_batch:
        out = 0
        x,y = model.input_shape(x,y)
        for i in range(N):
            out += softmax(model(x))
        out = out/N
        _, pred = torch.max(out, 1)
        correct += float((pred == y).sum())
        total += float(y.size(0))           
    return 100*correct/total, total - correct


def prediction(data_batch,model):
    softmax = nn.Softmax(dim = 1)
    correct = 0
    total = 0
    for x, y in data_batch:
        x,y = model.input_shape(x,y)
        out = softmax(model(x))
        _, pred = torch.max(out.data, 1)
        pred = Variable(pred)
        
        correct += float((pred == y).sum())
        total += float(y.size(0))
    return 100*correct/total, total - correct

def prediction_conv(data_batch, model, cuda):
    softmax = nn.Softmax(dim = 1)
    correct = 0
    total = 0
    for x, y in data_batch:
        x = Variable(x[:, None, :, :])
        y = Variable(y)
        if cuda:
            x, y = x.cuda(), y.cuda()

        out = softmax(model(x))
        _, pred = torch.max(out.data, 1)
        pred = Variable(pred)

        correct += float((pred == y).sum())
        total += float(y.size(0))
    return 100*correct/total, total - correct    

   
def batch_accuracy(data_batch,model, cuda): 
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(data_batch):
        targets = targets.squeeze()
        if cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)
        total += targets.size(0)

        _, pred = torch.max(outputs.data, 1)
        pred = Variable(pred)
        correct += float((pred == targets).sum())

    return 100*correct/total, total - correct



def batch_loss(data_batch,model,criterion): 
    loss = 0
    total = 0
    for batch_idx, (images,labels) in enumerate(data_batch):
#     for images, labels in data_batch:
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)  
        outputs = model(images)
        loss += criterion(outputs, labels)
        total += labels.size(0)
        
    return loss.data[0]/total


def linear_ini(LL,initialization):
    '''
    inputs : linear layer (LL) and the initialization
    output : linear layer with the chosen initialization
    '''
    if initialization == 'zero':
        LL.weight.data = nn.init.constant(LL.weight.data, 0)
        LL.bias.data = nn.init.constant(LL.bias.data, 0)
    
    if initialization == 'normal':
        LL.weight.data = nn.init.normal(LL.weight.data, 0,1)
        LL.bias.data = nn.init.normal(LL.bias.data, 0,1)

    if initialization == 'glorot':
        LL.weight.data = nn.init.xavier_uniform(LL.weight.data, gain=1)
        # that is important, see paper. 
        LL.bias.data = nn.init.constant(LL.bias.data, 0)
    if initialization == 'default': 
        pass
    return LL
