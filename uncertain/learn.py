# License: BSD
# Author: David Durst - based on http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html by Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torchsample.samplers import StratifiedSampler
import numpy as np
import time
import copy
from uncertain.uncertainCrossEntropyLoss import UncertainCrossEntropyLoss
import sys
import os

# inputs should be directory_of_data number_positive_examples output_file model_output_dir
data_dir = sys.argv[1]
num_training_str = sys.argv[2]
output_file = sys.argv[3]
model_output_dir = sys.argv[4]

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'val']}

#https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

weights = make_weights_for_balanced_classes(dsets['train'].imgs, len(dsets['train'].classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

dset_loaders = {}
dset_loaders['train'] = torch.utils.data.DataLoader(dsets['train'].imgs, batch_size=15, shuffle=False,
                                           sampler=sampler, num_workers=5, pin_memory=True)

dset_loaders['val'] =  torch.utils.data.DataLoader(dsets['val'].imgs, batch_size=15, shuffle=True,
                                           num_workers=5, pin_memory=True)

#dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=15,
#                                               shuffle=False, num_workers=4,
#                                                sampler=StratifiedSampler)
#                for x in ['train', 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = torch.cuda.is_available()

# Get a batch of training data
inputs, classes = next(iter(dset_loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

def classIndexToProbability(classIdx, class_to_idx_map):
    idx_to_class = {v: k for k, v in class_to_idx_map.items()}
    return [float(x) for x in idx_to_class[classIdx].split(",")]

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=125):
    since = time.time()

    best_model = model
    best_acc = 0.0
    final_layer_weights_last_iteration = model.fc.weight.clone()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labelIndices = data

                labelProbabilities = torch.FloatTensor([classIndexToProbability(index, dsets[phase].class_to_idx) for index in labelIndices])

                # wrap them in Variable
                if use_gpu:
                    inputs, labelProbabilitiesVar = Variable(inputs.cuda()), \
                        Variable(labelProbabilities.cuda())
                else:
                    inputs, labelProbabilitiesVar = Variable(inputs), Variable(labelProbabilities)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labelProbabilitiesVar)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # the correct label is the one with greatest probability
                _, labels = torch.max(labelProbabilitiesVar, dim=1)
                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print("last layer weights:")
        print(model.fc.weight)
        print('{:.7f}: sum of abs of difference in weights'.format(
            (final_layer_weights_last_iteration - model.fc.weight).abs().sum().data[0]))
        final_layer_weights_last_iteration = model.fc.weight.clone()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return (best_model, best_acc)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = UncertainCrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=0.001, momentum=0.9)

model_ft, best_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=125)

with open(output_file, 'a') as f:
    f.write(data_dir + "," + num_training_str + "," + str(best_acc) + "\n")

torch.save(model_ft.state_dict(), model_output_dir + "/" + num_training_str)