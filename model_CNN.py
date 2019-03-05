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

C = 64
rows = 72 # height
columns = 108 # width
p = 0 # padding
s = 1 # stride
krow = 1 
kcol = 5

outrows = (rows - krow + 2*p)/s + 1
outcols = (columns - kcol + 2*p)/s + 1


# class Single_IMU(nn.Module):

#     def __init__(self, num_channels=3, kernel=(1,5), pool=(1,2), num_classes=12):
#         super(Single_IMU, self).__init__()
#         self.conv1 = nn.Conv2d(num_channels, C, kernel, stride=1, padding=(0,0)) # try no padding first, column padding - padding=(0,1)
#         self.conv2 = nn.Conv2d(C, C, kernel, stride=1, padding=(0,0)) # try no padding first, column padding - padding=(0,1)
#         self.pool1 = nn.MaxPool2d(pool)
        
#         self.drop1 = nn.Dropout(0.5)
        
#         self.fc1 = nn.Linear(21*72*C, 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, num_classes)
        
#     def forward(self, X_imu1):
#         X_imu1 = F.relu(self.conv1(X_imu1))
#         X_imu1 = F.relu(self.conv2(X_imu1))
#         X_imu1 = self.pool1(X_imu1)
        
#         X_imu1 = F.relu(self.conv2(X_imu1))
#         X_imu1 = F.relu(self.conv2(X_imu1))
#         X_imu1 = self.pool1(X_imu1)

#         # print("After second pooling, shape is: {}".format(X_imu1.size()))
        
#         # 1st fully connected
#         X_imu1 = self.drop1(X_imu1)
#         X_imu1 = X_imu1.reshape(-1, 21*72*C)
#         # X_imu1 = X_imu1.reshape(X_imu1.size(0), -1)
#         X_imu1 = F.relu(self.fc1(X_imu1))
        
#         # 2nd fully connected
#         X_imu1 = self.drop1(X_imu1)
#         X_imu1 = F.relu(self.fc2(X_imu1))
        
#         # 3rd fully connected
#         X_imu1 = self.fc3(X_imu1)
        
#         return X_imu1  # logits 


class Single_Branch(nn.Module):

    def __init__(self, num_channels=1, kernel=(1,5), pool=(1,2), num_classes=12):
        super(Single_Branch, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, C, kernel_size=kernel, stride=1, padding=(0,0)),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=kernel, stride=1, padding=(0,0)),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=kernel, stride=1, padding=(0,0)),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=kernel, stride=1, padding=(0,0)),
            nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(21*72*C, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        
    def forward(self, X_imu1):
        out = self.layer1(X_imu1)
        out = self.layer2(out)
        out = out.reshape(-1, 21*72*C) # out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out # logits

# transform for the training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((72, 108)), # original size: (288, 432), resized to 25% of original size
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
        # path should be something like /data/mark/NetworkDatasets/pamap2/Train/
        X = Image.open(self.path + ID + '.jpg')

        y = self.labels[index]
    
        if self.transform:
            X = self.transform(X)

        return X, y



# CUDA for PyTorch
# use_cuda = torch.cuda.is_available()
# device = torch.device("cpu")
# device = torch.device("cuda" if use_cuda else "cpu")
# cudnn.benchmark = True # optimal set of algorithms

# Define dataloader parameters in dict
dlparams = {'batch_size': 50,
          'shuffle': True,
          'num_workers': 4} #how many?
epochs = 12

dataset_train = pd.read_csv('/data/mark/NetworkDatasets/pamap2_cnn/Train/figure_labels.csv', ',', header=0)
# dataset_train = pd.read_csv('D:/Fourth Year/FYP/PAMAP2_Dataset/Sample_dataset/Train/figure_labels.csv', ',', header=0)
# dataset_train = pd.read_csv('/home/mark/Repo/FYP_HAR/Sample_dataset/Train/figure_labels.csv', ',', header=0)

# Path to figures of all IMUs together (for training set)
train_path_imu1 = '/data/mark/NetworkDatasets/pamap2_cnn/Train/'
# train_path_imu1 = 'D:/Fourth Year/FYP/PAMAP2_Dataset/Sample_dataset/Train/'
# train_path_imu1 = '/home/mark/Repo/FYP_HAR/Sample_dataset/Train/'


training_set_imu1 = Dataset(dataset_train, train_path_imu1, train_transform)
train_loader_imu1 = DataLoader(training_set_imu1, batch_size=50, num_workers=4, shuffle=True)

#Validation data:
dataset_validation = pd.read_csv('/data/mark/NetworkDatasets/pamap2_cnn/Validation/figure_labels.csv', ',', header=0)
# dataset_validation = pd.read_csv('D:/Fourth Year/FYP/PAMAP2_Dataset/Sample_dataset/Validation/figure_labels.csv', ',', header=0)
# dataset_validation = pd.read_csv('/home/mark/Repo/FYP_HAR/Sample_dataset/Validation/figure_labels.csv', ',', header=0)

Validation_path_imu1 = '/data/mark/NetworkDatasets/pamap2_cnn/Validation/'
# Validation_path_imu1 = '/home/mark/Repo/FYP_HAR/Sample_dataset/Validation/'

Validation_set_imu1 = Dataset(dataset_validation, Validation_path_imu1, valid_transform)
Validation_loader_imu1 = DataLoader(Validation_set_imu1, batch_size=50, num_workers=4, shuffle=False)




class AverageBase(object):
    
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
       
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value
    
    
class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """
    
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count
        
    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    
    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value



model = Single_Branch()
model.to(device)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
# optimizer = optim.RMSprop(model.parameters(), lr=0.00001, alpha=0.95)




def save_checkpoint(optimizer, model, epoch, filename):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch



#!mkdir -p checkpoints

# count = 0
# for batch, targets in train_loader_imu1:
#     if count > 1:
#         break
#     mean = batch.mean()
#     std_dev = batch.std()
#     count += 1


# print(mean) # 0.9671
# print(std_dev) # 0.0596
# # print("mean = ".format(mean))
# # print("std_dev = ".format(std_dev))


def train(optimizer, model, num_epochs, first_epoch=1):

    # # Code for debugging
    # file1 = '/home/mark/predictions1.csv'
    # test_csv1 = open(file1, 'w')
    # writer1 = csv.writer(test_csv1)
    # writer1.writerow(['Model_Prediction', 'Actual_Activity'])

    # file2 = '/home/mark/predictions2.csv'
    # test_csv2 = open(file2, 'w')
    # writer2 = csv.writer(test_csv2)
    # writer2.writerow(['Model_Prediction', 'Actual_Activity'])
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []
    
        

    for epoch in range(first_epoch, first_epoch + num_epochs):
        print('Epoch', epoch)

        # train phase
        model.train()

        train_loss = MovingAverage()

        y_pred_train = []

        for batch, targets in train_loader_imu1:
            # Move the training data to the GPU
            batch = batch.to(device)
            targets = targets.to(device)

            # clear previous gradient computation
            optimizer.zero_grad()

            # forward propagation
            predictions = model(batch)

            # calculate the loss
            loss = criterion(predictions, targets)

            # backpropagate to compute gradients
            loss.backward()

            # update model weights
            optimizer.step()

            # update average loss
            train_loss.update(loss)

            # save training predictions
            y_pred_train.extend(predictions.argmax(dim=1).cpu().numpy())

            # update progress bar
            # progress.update(batch.shape[0], train_loss)

        print('Training loss:', train_loss)
        train_losses.append(train_loss.value)

        y_pred_train = torch.tensor(y_pred_train, dtype=torch.int64)
        train_labels_tensor = torch.from_numpy(training_set_imu1.labels)
        accuracy_train = torch.mean((y_pred_train == train_labels_tensor).float())
        print('Training accuracy: {:.4f}%'.format(float(accuracy_train) * 100))


        # validation phase
        model.eval()

        valid_loss = RunningAverage()

        # keep track of predictions
        y_pred = []

        correct = 0
        total = 0

        # We don't need gradients for validation, so wrap in 
        # no_grad to save memory
        with torch.no_grad():

            for batch, targets in Validation_loader_imu1:

                # Move the training batch to the GPU
                batch = batch.to(device)
                targets = targets.to(device)

                # forward propagation
                predictions = model(batch)

                # calculate the loss
                loss = criterion(predictions, targets)

                # update running loss value
                valid_loss.update(loss)

                # save predictions
                y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

                # # Code for debugging
                # if epoch == 1:
                #     writer1.writerow([(predictions.argmax(dim=1).cpu().numpy()), targets])

                # if epoch == 6:
                #     writer2.writerow([(predictions.argmax(dim=1).cpu().numpy()), targets])


                # y_pred2 = torch.max(predictions.data, 1)
                # total += targets.size(0)
                # targets_tensor = torch.from_numpy(targets)
                # correct += (y_pred2 == targets_tensor).sum().item()

        print('Validation loss:', valid_loss)
        valid_losses.append(valid_loss.value)

        # Calculate validation accuracy
        y_pred = torch.tensor(y_pred, dtype=torch.int64)
        valid_labels_tensor = torch.from_numpy(Validation_set_imu1.labels)
        accuracy = torch.mean((y_pred == valid_labels_tensor).float())
        print('Validation accuracy: {:.4f}%'.format(float(accuracy) * 100))

        # Save a checkpoint
        checkpoint_filename = '/home/mark/checkpoints/CNNDataset-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
    
    return train_losses, valid_losses, y_pred


train_losses, valid_losses, y_pred = train(optimizer, model, num_epochs=12)
