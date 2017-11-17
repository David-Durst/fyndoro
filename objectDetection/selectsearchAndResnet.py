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
output_dir = sys.argv[3]

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

model = model.cuda()

# take in the bounds from selective search (left, top, width, height)
# and crop the image using them
def cropImageUsingBounds(image, bounds):
    # cropCoordinates is in format left, upper, right, lower
    # which pillow's crop wants
    cropCoordinatesPillow = (bounds[0], bounds[1],
                             bounds[0] + bounds[2],
                             bounds[1] + bounds[3])

    return image.crop(cropCoordinatesPillow)


outputs = []
i = 0
for dataPoint in dset:
    print("Working on element " + str(i) + " of " + len(dset))
    image, labelIndices = dataPoint
    # just using default parameters, will tune later
    img_lbl, regions = selectivesearch.selective_search(numpy.asarray(image), scale=500, sigma=0.9, min_size=10)
    imageRegionsAsTensors = []
    for region in regions:
        # skip empty regions
        if region['rect'][2] == 0 or region['rect'][3] == 0:
            continue
        # skip regions of fewer than 2000 pixels
        if region['size'] < 2000:
            continue

        imageRegion = cropImageUsingBounds(region['rect'])
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
    # take the region with the max probability of being the desired
    # class
    # max(0) gives the indices and values of the max in the first
    # dimension fo tensor, which is max probability of being in
    # each category
    mostLikely = classProbabilityTensor.max(0)
    # first [0] gives the probabilities. skip if p < 0.9
    # second [0] gives the index of the element with the max probability
    # of being the first class (0 indexing)
    if mostLikely[0][0] < 0.9:
        continue
    # [1] gives the indices instead of the probabilities
    indexOfMostLikely = classProbabilityTensor.max(0)[1][0]
    fileName = os.path.basename(dset.imgs[i][0])
    cropImageUsingBounds(image, regions[indexOfMostLikely]['rect']).save(output_dir + "/" + fileName)