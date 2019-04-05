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

import sys
sys.path.append('..')

import plot_confusion_matrix as con

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
            self.data = X_test.transpose()
            self.labels = y_test

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
    l = int(y_test[jj+int(100/2)])
    img_labels.append(l)
    # img_labels[ctr] = l
    ctr += 1

print("length of img_labels = {}".format(len(img_labels)))

val_lab = np.array(img_labels)

# Validation data:
test_set = Dataset(dataset_pickle, valid_transform, train=False)
test_loader = DataLoader(test_set, batch_size=50, num_workers=4, shuffle=False)
print(test_set.img_labels[:])





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


PATH = '/data/mark/saved_models/baseline/cnn_imu.pt'


# # Model class must be defined somewhere
# model = torch.load(PATH)
# model.eval()


device = torch.device("cuda")
model = CNN_IMU_HR()
model.load_state_dict(torch.load(PATH))
model.to(device)

def test(model=model, test_loader=test_loader):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    best_y_pred = []


    valid_loss = RunningAverage()

    # keep track of predictions
    y_pred = []

    correct = 0
    total = 0

    # We don't need gradients for validation, so wrap in 
    # no_grad to save memory
    with torch.no_grad():

        for batch_imu1, batch_imu2, batch_imu3, batch_HR, targets in test_loader:

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

    print('Validation loss:', valid_loss)
    valid_losses.append(valid_loss.value)

    y_pred_arr = y_pred


    # Calculate validation accuracy
    y_pred = torch.tensor(y_pred, dtype=torch.int64)
    # valid_labels_tensor = torch.from_numpy(Validation_set.img_labels)
    valid_labels_tensor = torch.from_numpy(val_lab)
    # a = (y_pred == valid_labels_tensor)
    accuracy = torch.mean((y_pred == valid_labels_tensor).float())
    val_accuracy = float(accuracy) * 100
    print('Validation accuracy: {:.4f}%'.format(float(accuracy) * 100))


    y_true = np.asarray(val_lab)
    wf1 = con.getF1(y_true, y_pred_arr)

    wf1_percent = float(wf1) * 100
    print('Weighted F1: {:.4f}%'.format(float(wf1) * 100))

    return y_true, y_pred_arr, val_accuracy, wf1_percent


if __name__ == '__main__':

    class_names = con.class_names1

    y_true, y_pred_arr, val_accuracy, wf1_percent = test()
    figcon, zx = con.plot_confusion_matrix(y_true, y_pred_arr, classes=class_names, normalize=True,
                        title='Confusion Matrix - Overall Accuracy = {:.4f}%'.format(val_accuracy))

    figcon.savefig('/home/mark/Repo/FYP_HAR/Baseline/Confusion_graphs/confusion_cnn_imu_test.jpg')

    print('Done')