import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),			
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),			
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
			nn.Dropout(p=0.5),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
			nn.Dropout(p=0.5),
		
			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
			nn.Dropout(p=0.5),

			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
			nn.Dropout(p=0.5),

			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
			nn.Dropout(p=0.5))


		self.classifier = nn.Sequential(
			nn.Linear(2048, 512),
			nn.BatchNorm1d(512),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(512, 64),
			nn.BatchNorm1d(64),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(64, 2))

	def select_layer(self):
		conv_all = [self.features[0], self.features[3], 
                    self.features[8], self.features[11], 
                    self.features[16], self.features[19], self.features[22], 
                    self.features[27], self.features[30], self.features[33], self.features[36], 
                    self.features[41], self.features[44], self.features[47], self.features[50]]
		return conv_all

	def forward(self, x):
		out = self.features(x)
		out = self.classifier(out.view(out.size(0),-1))
		

		return out

class Classifier_pretrained(nn.Module):
	def __init__(self):
		super(Classifier_pretrained, self).__init__()

		vgg19_bn = models.vgg19_bn(pretrained=True)
		self.features = nn.Sequential(*list(vgg19_bn.children())[0])

		self.classifier = nn.Sequential(
			nn.Linear(2048, 512),
			nn.BatchNorm1d(512),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(512, 64),
			nn.BatchNorm1d(64),
			nn.Dropout(p=0.5),
			nn.ReLU(),
			nn.Linear(64, 2))

	def forward(self, x):
		out = self.features(x)
		out = self.classifier(out.view(out.size(0),-1))
		
		return out