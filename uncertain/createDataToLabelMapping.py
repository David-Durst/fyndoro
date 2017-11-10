# License: BSD
# Author: David Durst - based on http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html by Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
import os
import json

# inputs should be directory_of_data number_positive_examples output_file model_output_dir
data_dir = sys.argv[1]
model_input_location = sys.argv[2]
output_file = sys.argv[3]

# Data augmentation and normalization for training
# do transforms that are normally just for validation
# for all data
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
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

use_gpu = torch.cuda.is_available()

model = models.resnet18(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
#model.load_state_dict(torch.load(model_input_location))

# this produces the same model but with the last, fully connected layer removed
# so that the embedding of the image into a 512 dimension space is outputted
embedding_model = nn.Sequential(*list(model.children())[:-1])

if use_gpu:
    model = model.cuda()
    embedding_model = embedding_model.cuda()

outputs = {}
for phase in ['train', 'val']:
    print("In phase " + phase)
    phaseLen = str(len(dsets[phase]))
    i = 0
    outputs[phase] = []
    for data in dsets[phase]:
        print("Working on element " + str(i) + " of " + phaseLen + " in phase " + phase)
        inputs, labelIndices = data
        inputs = inputs.unsqueeze(0)
        # wrap them in Variable
        if use_gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        # based on __getitem__ implementation of datasets.ImageLoader, imgs index matches that of items
        classProbabilityTensor = F.softmax(model(inputs)).data
        # view -1 to remove the many, unnecessary dimensions that
        # have 1 value
        embeddingTensor = embedding_model(inputs).data.view(-1)
        if use_gpu:
            resultsList = classProbabilityTensor.cpu().numpy()[0].tolist()
            embeddingList = embeddingTensor.cpu().numpy().tolist()
        else:
            resultsList = classProbabilityTensor.numpy()[0].tolist()
            embeddingList = embeddingTensor.numpy().tolist()
        resultsList.insert(0, dsets[phase].imgs[i][0])
        resultsList.append(embeddingList)
        outputs[phase].append(resultsList)
        i += 1

with open(output_file, 'w') as f:
    json.dump(outputs, f, sort_keys=True, indent=2, separators=(',', ': '))