from PIL import Image
import selectivesearch
import numpy

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
# if this is empty, the default model will be used
model_input_location = sys.argv[2]
output_file = sys.argv[3]

# Data augmentation and normalization for training
# do transforms that are normally just for validation
# for all data
data_transforms = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dset = datasets.ImageFolder(data_dir)

use_gpu = torch.cuda.is_available()

model = models.resnet18(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
# if model_input_location is empty, use default weights
if model_input_location != "":
    model.load_state_dict(torch.load(model_input_location))

outputs = []
for dataPoint in dset:
    #print("Working on element " + str(i) + " of " + phaseLen + " in phase " + phase)
    image, labelIndices = dataPoint
    # just using default parameters, will tune later
    img_lbl, regions = selectivesearch.selective_search(numpy.asarray(image), scale=500, sigma=0.9, min_size=10)
    imageRegionsAsTensors = []
    for region in regions:
        # cropBounds is in format left, top, width, height
        cropBoundsSelectSearch = region['rect']
        # cropCoordinates is in format left, upper, right, lower
        # which pillow's crop wants
        cropCoordinatesPillow = (cropBoundsSelectSearch[0], cropBoundsSelectSearch[1],
                                 cropBoundsSelectSearch[0] + cropBoundsSelectSearch[2],
                                 cropBoundsSelectSearch[1] + cropBoundsSelectSearch[3])

        imageRegion = image.crop(cropCoordinatesPillow)
        imageAsTensorForEval = data_transforms(imageRegion)
        imageRegionsAsTensors.append(imageAsTensorForEval)

    imageRegionsAsOneTensor = torch.stack(imageRegionsAsTensors)
    # wrap them in Variable
    if use_gpu:
        inputs = Variable(imageRegionsAsOneTensor.cuda())
    else:
        inputs = Variable(imageRegionsAsOneTensor)
    # based on __getitem__ implementation of datasets.ImageLoader, imgs index matches that of items
    classProbabilityTensor = F.softmax(model(inputs)).data
    # view -1 to remove the many, unnecessary dimensions that
    # have 1 value
    if use_gpu:
        classProbability = classProbabilityTensor.cpu().numpy()[0]
    else:
        classProbability = classProbabilityTensor.numpy()[0]
