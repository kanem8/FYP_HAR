# make sure to enable GPU acceleration!
device = 'cuda'

import numpy as np
from skimage import io, transform
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torchvision.transforms as transforms

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import pandas as pd
# from itertools import izip
import csv
import pickle

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
            # nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.Conv2d(C, C, kernel_size=kernel, stride=1, padding=(0,0)),
            # nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=pool)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=kernel, stride=1, padding=(0,0)),
            # nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.Conv2d(C, C, kernel_size=kernel, stride=1, padding=(0,0)),
            # nn.BatchNorm2d(C),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool),
            nn.Dropout(0.5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(C*40*19, 512),
            # nn.Linear(4*100*C, 512),
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
        # print(out.shape)
        # out = out.reshape(out.size(0), -1)
        out = out.reshape(-1, C*40*19) # out = out.reshape(out.size(0), -1)
        # print(out.shape)
        # out = out.reshape(-1, 4*100*C) 
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out # logits

# transform for the training data
train_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((72, 108)), # original size: (288, 432), resized to 25% of original size
    # transforms.Resize((100, 40)), # original size: (288, 432), resized to 25% of original size
    # transforms.ToTensor(),
    # transforms.functional.to_tensor(),

    # transforms.Normalize([0.9671], [0.0596]) # unsure how to normalize the tensor correctly
])

# use the same transform for the validation data
valid_transform = train_transform


def window(i, x_data, y_label, window_size):
    X = x_data[:,i:(i+window_size)]
    y = int(y_label[i+int(window_size/2)])


    # if (i+window_size) > y_label.shape[0]:
    #     # print("i = {}".format(i))
    #     # print("y_label.shape[0] = {}".format(y_label.shape[0]))
    #     y = int(y_label[i])
    # else:
    #     y = int(y_label[i+int(window_size/2)])


    return X, y




from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, pickle_file, transform, train=True, window_size=100, frame_shift=22): #used to be passed: list_IDs, labels, 
        'Initialization'

        self.pickle_file = pickle_file
        self.train = train
        self.transform = transform
        self.window_size = window_size
        self.frame_shift = frame_shift
        # self.img_labels = np.empty((0))
        # self.img_labels = np.array()
        # self.img_labels = []

        with open(pickle_file, 'rb') as f:
            [(X_train, y_train), (X_val, y_val), (X_test, y_test)] = pickle.load(f)

        if self.train:
            self.data = X_train.transpose() # transpose so that sensor channels are rows and samples are columns, can now apply sliding window
            self.labels = y_train
        else:
            self.data = X_val.transpose()
            self.labels = y_val

        images = int((len(self.labels) - self.window_size)/(self.frame_shift))
        self.img_labels = np.zeros((images), dtype=np.int64)

        indexes = list(range(0, images, self.frame_shift))
        ctr = 0
        for j in indexes:
            l = int(self.labels[j+int(self.window_size/2)])
            self.img_labels[ctr] = l
            ctr += 1

        print("shape of data: {}".format(self.data.shape))
        print("shape of labels: {}".format(self.labels.shape))

  def __len__(self):
        'Denotes the total number of samples'
        return int((len(self.labels) - self.window_size)/(self.frame_shift))

  def __getitem__(self, index):
        'Generates one sample of data'
        
        i = index*(self.frame_shift)
        X, y = window(i, self.data, self.labels, self.window_size)
    
        if self.transform:
            X = torch.from_numpy(X).float()
            X = X.unsqueeze(dim=0)
            # X = self.transform(X)
        
        # self.img_labels = self.img_labels.append(y)

        # self.img_labels = np.append(self.img_labels, y)
        # self.img_labels[index] = y



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


dataset_pickle = '/data/mark/NetworkDatasets/baseline/pamap2.data'

# Training data
training_set = Dataset(dataset_pickle, train_transform, train=True)
train_loader = DataLoader(training_set, batch_size=50, num_workers=4, shuffle=True)
print(training_set.img_labels[0:10])

# Validation data:
Validation_set = Dataset(dataset_pickle, valid_transform, train=False)
Validation_loader = DataLoader(Validation_set, batch_size=50, num_workers=4, shuffle=False)
print(Validation_set.img_labels[:])

indexes = list(range(0, 4123))
for j in indexes: 
    if (j % 100 != 0):
        print()
    print(Validation_set.img_labels[j], end=' ')





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



# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         init.orthogonal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)

#     # elif isinstance(m, nn.BatchNorm2d):
#     #     init.orthogonal_(m.weight.data)
#     #     init.orthogonal_(m.bias.data)

#     elif isinstance(m, nn.Linear):
#         init.orthogonal_(m.weight.data)
#         if m.bias is not None:
#             init.normal_(m.bias.data)


model = Single_Branch()

# model.apply(weights_init)

model.to(device)


# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.95)




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
# for batch, targets in train_loader:
#     if count > 1:
#         break
#     mean = batch.mean()
#     std_dev = batch.std()
#     count += 1


# print(mean) # 0.9671
# print(std_dev) # 0.0596
# # print("mean = ".format(mean))
# # print("std_dev = ".format(std_dev))

# figure_csv = "~/Repo/FYP_HAR/Baseline/predictions.csv"
# file_csv = open(figure_csv, 'w')
# writer = csv.writer(file_csv)
# writer.writerow(['predictions', 'targets', 'equal'])


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

        for batch, targets in train_loader:
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
        train_labels_tensor = torch.from_numpy(training_set.img_labels)
        accuracy_train = torch.mean((y_pred_train == train_labels_tensor).float())
        print('Training accuracy: {:.4f}%'.format(float(accuracy_train) * 100))


        print(y_pred_train.size())
        print(train_labels_tensor.size())

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

            for batch, targets in Validation_loader:

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
        valid_labels_tensor = torch.from_numpy(Validation_set.img_labels)
        a = (y_pred == valid_labels_tensor)
        accuracy = torch.mean((y_pred == valid_labels_tensor).float())
        print('Validation accuracy: {:.4f}%'.format(float(accuracy) * 100))
        
        print(y_pred.size())
        print(valid_labels_tensor.size())
        print(a.size())

        print(y_pred)
        print(valid_labels_tensor)
        print(a)

        # indexes = list(range(0, 4123))
        # for j in indexes:
        #     print(y_pred[j], end=' '),
        #     print(valid_labels_tensor[j], end=' '),
        #     print(a[j])

        # writer.writecolumn 

        # Save a checkpoint
        checkpoint_filename = '/data/mark/NetworkDatasets/baseline/checkpoints/Baseline-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
    
    return train_losses, valid_losses, y_pred


train_losses, valid_losses, y_pred = train(optimizer, model, num_epochs=1)