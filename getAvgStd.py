# make sure to enable GPU acceleration!
device = 'cuda'

import numpy as np
from skimage import io, transform
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import pandas as pd
# from itertools import izip
import csv

print('PyTorch version:', torch.__version__)

torch.cuda.is_available()

torch.version.cuda

# Set random seed for reproducability
torch.manual_seed(271828)
np.random.seed(271728)


# transform for the training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((72, 108)), # original size: (288, 432), resized to 25% of original size
    # transforms.Resize((100, 40)), # original size: (288, 432)
    # transforms.Resize((40, 100)), # original size: (288, 432)
    transforms.ToTensor(),
    # transforms.Normalize([0.9671], [0.0596]) # unsure how to normalize the tensor correctly
])

# use the same transform for the validation data
valid_transform = train_transform

from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataframe, path, transform): #used to be passed: list_IDs, labels, 
        'Initialization'
#         self.labels = labels
#         self.list_IDs = list_IDs

        self.list_IDs = dataframe.iloc[:,0:1].values.reshape(-1)
        self.labels = dataframe.iloc[:,1:].values.reshape(-1)
        print("shape of IDs: {}".format(self.list_IDs.shape))
        print("shape of labels: {}".format(self.labels.shape))
        self.path = path
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
#         X = torch.load('data/' + ID + '.pt')
        # path should be something like /data/mark/NetworkDatasets/pamap2/Train/IMU_1Hand/
        X = Image.open(self.path + ID + '.jpg')

#         y = self.labels[ID] ID??
        y = self.labels[index]
    
        if self.transform:
            X = self.transform(X)

        return X, y


dataset_train = pd.read_csv('/data/mark/NetworkDatasets/pamap2_HR/Train/figure_labels.csv', ',', header=0)

train_path_imu1 = '/data/mark/NetworkDatasets/pamap2_HR/Train/IMU_1Hand/'
train_path_imu2 = '/data/mark/NetworkDatasets/pamap2_HR/Train/IMU_2Chest/'
train_path_imu3 = '/data/mark/NetworkDatasets/pamap2_HR/Train/IMU_3Ankle/'
train_path_HR = '/data/mark/NetworkDatasets/pamap2_HR/Train/HR_Sensor/'


training_set_imu1 = Dataset(dataset_train, train_path_imu1, train_transform)
train_loader_imu1 = DataLoader(training_set_imu1, batch_size=50, num_workers=4, shuffle=True)

training_set_imu2 = Dataset(dataset_train, train_path_imu2, train_transform)
train_loader_imu2 = DataLoader(training_set_imu2, batch_size=50, num_workers=4, shuffle=True)

training_set_imu3 = Dataset(dataset_train, train_path_imu3, train_transform)
train_loader_imu3 = DataLoader(training_set_imu3, batch_size=50, num_workers=4, shuffle=True)

training_set_HR = Dataset(dataset_train, train_path_HR, train_transform)
train_loader_HR = DataLoader(training_set_HR, batch_size=50, num_workers=4, shuffle=True)


#Validation data:
# dataset_validation = pd.read_csv('/data/mark/NetworkDatasets/pamap2/Validation/figure_labels.csv', ',', header=0)

# Validation_path_imu1 = '/data/mark/NetworkDatasets/pamap2/Validation/IMU_1Hand/'
# Validation_path_imu2 = '/data/mark/NetworkDatasets/pamap2/Validation/IMU_2Chest/'
# Validation_path_imu3 = '/data/mark/NetworkDatasets/pamap2/Validation/IMU_3Ankle/'

dataset_validation = pd.read_csv('/data/mark/NetworkDatasets/pamap2_HR/Validation/figure_labels.csv', ',', header=0)

Validation_path_imu1 = '/data/mark/NetworkDatasets/pamap2_HR/Validation/IMU_1Hand/'
Validation_path_imu2 = '/data/mark/NetworkDatasets/pamap2_HR/Validation/IMU_2Chest/'
Validation_path_imu3 = '/data/mark/NetworkDatasets/pamap2_HR/Validation/IMU_3Ankle/'
Validation_path_HR = '/data/mark/NetworkDatasets/pamap2_HR/Validation/HR_Sensor/'

Validation_set_imu1 = Dataset(dataset_validation, Validation_path_imu1, valid_transform)
Validation_loader_imu1 = DataLoader(Validation_set_imu1, batch_size=50, num_workers=4, shuffle=False)

Validation_set_imu2 = Dataset(dataset_validation, Validation_path_imu2, valid_transform)
Validation_loader_imu2 = DataLoader(Validation_set_imu2, batch_size=50, num_workers=4, shuffle=False)

Validation_set_imu3 = Dataset(dataset_validation, Validation_path_imu3, valid_transform)
Validation_loader_imu3 = DataLoader(Validation_set_imu3, batch_size=50, num_workers=4, shuffle=False)

Validation_set_HR = Dataset(dataset_validation, Validation_path_HR, valid_transform)
Validation_loader_HR = DataLoader(Validation_set_HR, batch_size=50, num_workers=4, shuffle=False)




count = 0
for batch, targets in train_loader_imu1:
    if count > 10:
        break
    mean = batch.mean()
    print("mean = {}".format(mean))
    std_dev = batch.std()
    print("std dev = {}".format(std_dev))
    count += 1


# print(mean) # 0.9671
# print(std_dev) # 0.0596
# print("mean = ".format(mean))
# print("std_dev = ".format(std_dev))