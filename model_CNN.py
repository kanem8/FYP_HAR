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

print('PyTorch version:', torch.__version__)

torch.cuda.is_available()

torch.version.cuda

# Set random seed for reproducability
torch.manual_seed(271828)
np.random.seed(271728)

C = 64

class Single_IMU(nn.Module):

    def __init__(self, num_channels=1, kernel=(1,5), pool=(1,2), num_classes=12):
        super(Single_IMU, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel, stride=1, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(pool)
        
        self.drop1 = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(108*288*C, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, X_imu1):
        X_imu1 = F.relu(self.conv1(X_imu1))
        X_imu1 = F.relu(self.conv2(X_imu1))
        X_imu1 = self.pool1(X_imu1)
        
        X_imu1 = F.relu(self.conv2(X_imu1))
        X_imu1 = F.relu(self.conv2(X_imu1))
        X_imu1 = self.pool1(X_imu1)

        X_imu1.print("After second pooling, shape is: {}".format(X_imu1.shape))
        
        # 1st fully connected
        X_imu1 = self.drop1(X_imu1)
        X_imu1 = X_imu1.reshape(-1, 108*288*C)
        X_imu1 = F.relu(self.fc1(X_imu1))
        
        # 2nd fully connected
        X_imu1 = self.drop1(X_imu1)
        X_imu1 = F.relu(self.fc2(X_imu1))
        
        # 3rd fully connected
        X_imu1 = self.fc3(X_imu1)
        
        return X_imu1  # logits 

# transform for the training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    #transform.Resize(()),
    transforms.ToTensor(),
    #transforms.Normalize([0.1307], [0.3081])
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

# dataset_train = pd.read_csv('/data/mark/NetworkDatasets/pamap2/Train/figure_labels.csv', ',', header=0)
# dataset_train = pd.read_csv('D:/Fourth Year/FYP/PAMAP2_Dataset/Sample_dataset/Train/figure_labels.csv', ',', header=0)
dataset_train = pd.read_csv('/home/mark/Repo/FYP_HAR/Sample_dataset/Train/figure_labels.csv', ',', header=0)


# train_path_imu1 = '/data/mark/NetworkDatasets/pamap2/Train/IMU_1Hand/'
# train_path_imu1 = 'D:/Fourth Year/FYP/PAMAP2_Dataset/Sample_dataset/Train/'
train_path_imu1 = '/home/mark/Repo/FYP_HAR/Sample_dataset/Train/'


training_set_imu1 = Dataset(dataset_train, train_path_imu1, train_transform)
train_loader_imu1 = DataLoader(training_set_imu1, batch_size=16, num_workers=4, shuffle=True)

#Validation data:
# dataset_validation = pd.read_csv('/data/mark/NetworkDatasets/pamap2/Validation/figure_labels.csv', ',', header=0)
# dataset_validation = pd.read_csv('D:/Fourth Year/FYP/PAMAP2_Dataset/Sample_dataset/Validation/figure_labels.csv', ',', header=0)
dataset_validation = pd.read_csv('/home/mark/Repo/FYP_HAR/Sample_dataset/Validation/figure_labels.csv', ',', header=0)

# Validation_path_imu1 = '/data/mark/NetworkDatasets/pamap2/Validation/IMU_1Hand/'
Validation_path_imu1 = '/home/mark/Repo/FYP_HAR/Sample_dataset/Validation/'

Validation_set_imu1 = Dataset(dataset_validation, Validation_path_imu1, valid_transform)
Validation_loader_imu1 = DataLoader(Validation_set_imu1, batch_size=16, num_workers=4, shuffle=False)




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




from IPython.display import HTML, display

class ProgressMonitor(object):
    """
    Custom IPython progress bar for training
    """
    
    tmpl = """
        <p>Loss: {loss:0.4f}   {value} / {length}</p>
        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>
    """

    def __init__(self, length):
        self.length = length
        self.count = 0
        self.display = display(self.html(0, 0), display_id=True)
        
    def html(self, count, loss):
        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))
        
    def update(self, count, loss):
        self.count += count
        self.display.update(self.html(self.count, loss))



model = Single_IMU()
model.to(device)


optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)


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


def train(optimizer, model, num_epochs=12, first_epoch=1):
    
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    valid_losses = []

    for epoch in range(first_epoch, first_epoch + num_epochs):
        print('Epoch', epoch)

        # train phase
        model.train()

        # create a progress bar
        progress = ProgressMonitor(length=len(training_set_imu1))

        train_loss = MovingAverage()

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

            # update progress bar
            progress.update(batch.shape[0], train_loss)

        print('Training loss:', train_loss)
        train_losses.append(train_loss.value)


        # validation phase
        model.eval()

        valid_loss = RunningAverage()

        # keep track of predictions
        y_pred = []

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

        print('Validation loss:', valid_loss)
        valid_losses.append(valid_loss.value)

        # Calculate validation accuracy
        y_pred = torch.Tensor(y_pred, dtype=torch._C.int64)
        accuracy = torch._C.mean((y_pred == Validation_set_imu1.labels).float())
        print('Validation accuracy: {:.4f}%'.format(float(accuracy) * 100))

        # Save a checkpoint
        checkpoint_filename = '/home/mark/checkpoints/sampleset-{:03d}.pkl'.format(epoch)
        save_checkpoint(optimizer, model, epoch, checkpoint_filename)
    
    return train_losses, valid_losses, y_pred


train_losses, valid_losses, y_pred = train(optimizer, model, num_epochs=12)