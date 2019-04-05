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
import pickle

print('PyTorch version:', torch.__version__)

torch.cuda.is_available()

torch.version.cuda

# Set random seed for reproducability
torch.manual_seed(271828)
np.random.seed(271728)

m = 4 # number of branches

C = 64
rows = 72 # height
columns = 108 # width
p = 0 # padding
s = 1 # stride
krow = 1 
kcol = 5

outrows = (rows - krow + 2*p)/s + 1
outcols = (columns - kcol + 2*p)/s + 1


class CNN_IMU_HR(nn.Module):

    def __init__(self, num_channels=1, kernel=(1,5), pool=(1,2), num_classes=12):
        super(CNN_IMU_HR, self).__init__()

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
            nn.Linear(C*13*19, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fcHR = nn.Sequential(
            nn.Linear(C*1*19, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512*m, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, num_classes)
        )
        
    def forward(self, X_imu1, X_imu2, X_imu3, X_HR):
        out1 = self.layer1(X_imu1)
        out1 = self.layer2(out1)
        out1 = out1.reshape(-1, C*13*19)
        out1 = self.fc1(out1)

        out2 = self.layer1(X_imu2)
        out2 = self.layer2(out2)
        out2 = out2.reshape(-1, C*13*19)
        out2 = self.fc1(out2)

        out3 = self.layer1(X_imu3)
        out3 = self.layer2(out3)
        out3 = out3.reshape(-1, C*13*19)
        out3 = self.fc1(out3)

        out4 = self.layer1(X_HR)
        out4 = self.layer2(out4)
        out4 = out4.reshape(-1, C*1*19)
        out4 = self.fcHR(out4)

        combined = torch.cat((out1, out2, out3, out4), dim=1)

        combined = self.fc2(combined)
        combined = self.fc3(combined)

        return combined # logits


# transform for the training data
train_transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    # transforms.Resize((72, 108)),
    transforms.ToTensor(),
    # transforms.Normalize([0.9671], [0.0596])
    # transforms.Normalize([0.1307], [0.3081])
])

# use the same transform for the validation data
valid_transform = train_transform

def window(i, x_data, y_label, window_size):
    X = x_data[:,i:(i+window_size)]
    y = int(y_label[i+int(window_size/2)])

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
        return (int((len(self.labels) - self.window_size)/(self.frame_shift)) + 1)

  def __getitem__(self, index):
        'Generates one sample of data'
        
        i = index*(self.frame_shift)
        X_HR, y = window(i, self.data[0:1,:], self.labels, self.window_size)
        X_imu1, y = window(i, self.data[1:14,:], self.labels, self.window_size)
        X_imu2, y = window(i, self.data[14:27,:], self.labels, self.window_size)
        X_imu3, y = window(i, self.data[27:40,:], self.labels, self.window_size)


        if self.transform:
            X_HR = torch.from_numpy(X_HR).float()
            X_HR = X_HR.unsqueeze(dim=0)
            X_imu1 = torch.from_numpy(X_imu1).float()
            X_imu1 = X_imu1.unsqueeze(dim=0)
            X_imu2 = torch.from_numpy(X_imu2).float()
            X_imu2 = X_imu2.unsqueeze(dim=0)
            X_imu3 = torch.from_numpy(X_imu3).float()
            X_imu3 = X_imu3.unsqueeze(dim=0)
            # X = self.transform(X)
        
        # self.img_labels = self.img_labels.append(y)

        # self.img_labels = np.append(self.img_labels, y)
        # self.img_labels[index] = y

        return X_imu1, X_imu2, X_imu3, X_HR, y

dataset_pickle = '/data/mark/NetworkDatasets/baseline/pamap2.data'

with open(dataset_pickle, 'rb') as f:
    [(X_train, y_train), (X_val, y_val), (X_test, y_test)] = pickle.load(f)

images = int(((y_val.shape[0]) - 100)/22)
# img_labels = np.zeros((images), dtype=np.int64)
img_labels = []

print("images = {}".format(images))

indexes = list(range(0, y_val.shape[0]-100, 22))
print("length of indexes = {}".format(len(indexes)))

ctr = 0
for jj in indexes:
    l = int(y_val[jj+int(100/2)])
    img_labels.append(l)
    # img_labels[ctr] = l
    ctr += 1

print("length of img_labels = {}".format(len(img_labels)))

val_lab = np.array(img_labels)

# Training data
training_set = Dataset(dataset_pickle, train_transform, train=True)
train_loader = DataLoader(training_set, batch_size=50, num_workers=4, shuffle=True)
print(training_set.img_labels[0:10])

# Validation data:
Validation_set = Dataset(dataset_pickle, valid_transform, train=False)
Validation_loader = DataLoader(Validation_set, batch_size=50, num_workers=4, shuffle=False)
print(Validation_set.img_labels[0:10])

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


model = CNN_IMU_HR()
model.to(device)


tot_epochs = 21
step = int(tot_epochs/3)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.95)
scheduler = optim.lr_scheduler.StepLR(optimizer, step, gamma=0.1)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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


def train(optimizer, model, num_epochs, first_epoch=1):

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
        scheduler.step()
        current_lr = get_lr(optimizer)
        print("Current learning rate = {}".format(current_lr))

        # train phase
        model.train()

        # create a progress bar
        # progress = ProgressMonitor(length=len(training_set_imu1))

        train_loss = MovingAverage()

        y_pred_train = []

        # for (batch, targets), (batch2, targets2), (batch3, targets3), (batch4, targets4) in zip(train_loader_imu1, train_loader_imu2, train_loader_imu3, train_loader_HR):
        for batch_imu1, batch_imu2, batch_imu3, batch_HR, targets in train_loader:

            # # Move the training data to the GPU
            # batch = batch.to(device)
            # targets = targets.to(device)
            # batch2 = batch2.to(device)
            # targets2 = targets2.to(device)            
            # batch3 = batch3.to(device)
            # targets3 = targets3.to(device)   
            # batch4 = batch4.to(device)
            # targets4 = targets4.to(device)    
            # 
            # Move the training data to the GPU
            batch_imu1 = batch_imu1.to(device)
            batch_imu2 = batch_imu2.to(device)
            batch_imu3 = batch_imu3.to(device)
            batch_HR = batch_HR.to(device)
            targets = targets.to(device)

            # import pdb; pdb.set_trace()


            # clear previous gradient computation
            optimizer.zero_grad()

            # forward propagation
            predictions = model(batch_imu1, batch_imu2, batch_imu3, batch_HR)

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

        # y_pred_train = torch.tensor(y_pred_train, dtype=torch.int64)
        # train_labels_tensor = torch.from_numpy(training_set.img_labels)
        # accuracy_train = torch.mean((y_pred_train == train_labels_tensor).float())
        # print('Training accuracy: {:.4f}%'.format(float(accuracy_train) * 100))


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

            for batch_imu1, batch_imu2, batch_imu3, batch_HR, targets in Validation_loader:

                # # Move the training data to the GPU
                # batch = batch.to(device)
                # targets = targets.to(device)
                # batch2 = batch2.to(device)
                # targets2 = targets2.to(device)            
                # batch3 = batch3.to(device)
                # targets3 = targets3.to(device)  
                # batch4 = batch4.to(device)
                # targets4 = targets4.to(device)  
                batch_imu1 = batch_imu1.to(device)
                batch_imu2 = batch_imu2.to(device)
                batch_imu3 = batch_imu3.to(device)
                batch_HR = batch_HR.to(device)
                targets = targets.to(device)

                # forward propagation
                predictions = model(batch_imu1, batch_imu2, batch_imu3, batch_HR)

                # calculate the loss
                loss = criterion(predictions, targets)

                # update running loss value
                valid_loss.update(loss)

                # save predictions
                y_pred.extend(predictions.argmax(dim=1).cpu().numpy())

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
        # valid_labels_tensor = torch.from_numpy(Validation_set.img_labels)
        valid_labels_tensor = torch.from_numpy(val_lab)
        # a = (y_pred == valid_labels_tensor)s
        accuracy = torch.mean((y_pred == valid_labels_tensor).float())
        print('Validation accuracy: {:.4f}%'.format(float(accuracy) * 100))

        # Save a checkpoint
        checkpoint_filename = '/data/mark/NetworkDatasets/baseline/checkpoints/CNN_IMU-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
    
    return train_losses, valid_losses, y_pred


train_losses, valid_losses, y_pred = train(optimizer, model, num_epochs=tot_epochs)