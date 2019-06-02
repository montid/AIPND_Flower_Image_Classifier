"""
This script will train a new network on a dataset and save the model as a checkpoint.
It prints out training loss, validation loss, and validation accuracy as the network trains.

Required Inputs:    - data_dir: ImageNet Dataset folder path (default="./flowers/")

Optional inputs:    - arch: choose architecture (vgg or densenet) - default = vgg
                    - hidden_units: customize hidden layers (default=512)
                    - learning_rate: customize lr (default=0.0005)
                    - droput: customize droput (default=0.4)
                    - epochs: customize epochs (default=5)
                    - gpu: gpu ON (default = cpu)
                    - save_dir: customize checkpoint.pth destination folder (default = "./")
"""

# Imports here
import time
import torch
from torch import nn
from torch import optim
#import torch.nn.functional as F
#from torchvision import datasets, models
#import json
#from collections import OrderedDict
import argparse
#import numpy as np
import myutils
import nn_model


# Command line arguments

parser = argparse.ArgumentParser(description='Train network arguments:')
parser.add_argument('data_dir', default="./flowers/", help='Path to dataset')
parser.add_argument('--arch', dest="arch", action="store", default="vgg", type = str, help='Choose architecture {vgg or densenet}')
parser.add_argument('--hidden_units', type=int, nargs='+', dest="hidden_units", default=512, help='If more than one layer put a space (i.e. 512 250)')
parser.add_argument('--learning_rate', type=float, dest="learning_rate", action="store", default=0.0005, help='Learning Rate')
parser.add_argument('--dropout', type=float, dest = "dropout", action = "store", default = 0.4, help='DropOut')
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=5, help='Epochs')
#parser.add_argument('--gpu', type=str, dest="gpu", action="store", default="gpu", help='Gpu')
parser.add_argument('--gpu', help='Use GPU for training instead of CPU', action='store_true')
parser.add_argument('--save_dir', type=str, dest="save_dir", action="store", default="./", help='Checkpoint destination folder')


# Hyperparameters
parcy = parser.parse_args()
data_dir = parcy.data_dir
arch = parcy.arch
hidden_layers = myutils.check_h(parcy.hidden_units)
learning_rate = parcy.learning_rate
drop = parcy.dropout
epochs = parcy.epochs
isgpu = parcy.gpu
save_dir = parcy.save_dir

output_size = 102

# Print Hyperparameters
print("{}data_dir:    {} {} ".format('\n','\t', data_dir))
print("arch:          {} {} ".format('\t', arch))
print("hidden_units:  {} {} ".format('\t',hidden_layers))
print("learning_rate: {} {} ".format('\t',learning_rate))
print("dropout:       {} {} ".format('\t',drop))
print("epochs:        {} {} ".format('\t',epochs))
print("save_dir:      {} {} ".format('\t',save_dir))

if isgpu:
    print("GPU:       {} {} {}".format('\t', 'ON', '\n'))
else:
    print("GPU:       {} {} {}".format('\t', 'OFF', '\n'))


# load datasets, trainloader and validloader
image_datasets, dataloaders = myutils.load_data(data_dir)

# load a pre-trained network
model, input_size = nn_model.load_pretrained_model(arch)

## Global use
# Set the device (using GPU or not)
device = torch.device("cuda:0" if isgpu is True else "cpu")

#create my feedforward classifier
classifier = nn_model.Network(input_size, output_size, hidden_layers, drop)

#substitute the model's classifier with this new classifier
#transfer learning connected here
model.classifier = classifier
#print(model)

#define criteria and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Move the model to the device available (cpu or CUDA)
model.to(device)

## Train It ##

# Define some variable
#epochs = 5
steps = 0
running_loss = 0
print_every = 102

since = time.time()

print("START TRAINING ...")
# get images and labels batch (64)
for epoch in range(epochs):
    model.train()
    for images, labels in dataloaders['train']:
        steps += 1
        # move them to CPU or CUDA (It depend on the device)
        images, labels = images.to(device), labels.to(device)
        # training loop
        optimizer.zero_grad() # reset to zero the gradients
        logps = model.forward(images) # log probabilities (output)
        loss = criterion(logps, labels) # error calculation (cost function)
        loss.backward() # backpropagation pass
        optimizer.step() # updating weights
        running_loss += loss.item() # keeping track of the training loss

        ## Validate It ##

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval() # evaluation inference mode which turns dropout off
            with torch.no_grad():

                for images, labels in dataloaders['valid']:
                    # move them to GPU or CUDA (It depends on the device)
                    images, labels = images.to(device), labels.to(device)
                    logps = model.forward(images) # log probabilities (output)
                    batch_loss = criterion(logps, labels) # error calculation (cost function)
                    valid_loss += batch_loss.item() # keeping track of the test loss

                    # calculate the accuracy
                    ps = torch.exp(logps) # transform the LogSoftmax output to get the actual probabilities
                    top_ps, top_class = ps.topk(1,dim=1) # first largest value in our ps along the columns
                    # comparing the one element in each row of top_class with each element in labels
                    # which returns 64 True/False boolean values for each row.
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                  f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")

            running_loss = 0
            model.train()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# set class_to_idx mapping for model
model.class_to_idx = image_datasets['train'].class_to_idx

nn_model.save_model_checkpoint(model, input_size, epochs, save_dir, arch, hidden_layers, learning_rate, drop, optimizer, output_size)
print("New network trained on a dataset and the model saved as a checkpoint")
