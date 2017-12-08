#from objectDetection.classActivationMapResnet import makeAndSaveToFileCamClassificationHeatmap, getLargestConnectComponentAsPILImage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import sys
import os
import spn
from experiment.models import vgg16_sp
import numpy as np
import cv2
import json

# inputs should be directory_of_data number_positive_examples output_file model_output_dir
data_dir = sys.argv[1]
# if this is empty, the default model will be used
model_input_location = sys.argv[2]
output_dir_class0 = sys.argv[3]
output_dir_class1 = sys.argv[4]
output_dir_skip = sys.argv[5]
label_map = sys.argv[6]
# how likely must a category be for it to be chosen
categoryThreshold = float(sys.argv[7])

class regionCoordinate:
    def __init__(self, x, y, w, h, scale):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.scale = scale

    def __str__(self):
        return "x: {0}, y: {1}, w: {2}, h: {3}, scale: {4}".format(self.x, self.y, self.w, self.h, self.scale)

# Data augmentation and normalization for training
# do transforms that are normally just for validation
# for all data
imageScales = [1000, 750, 500, 250] # how big will the whole image be in pixels
imageMaxScale = max(imageScales)


network_input_width = 250 # widths the network expeccts
regionCoordinates = [regionCoordinate(x, y, network_input_width, network_input_width, s) for s in imageScales for x in range(0, s, 250) for y in range(0, s, 250)]

data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dset = datasets.ImageFolder(data_dir)

use_gpu = torch.cuda.is_available()

model = vgg16_sp(3, pretrained=True)
checkpoint = torch.load(model_input_location)
model.load_state_dict(checkpoint['state_dict'])
#for param in model.parameters():
#    param.requires_grad = False
# if model_input_location is empty, use default weights

model = model.cuda()

# take in the bounds from selective search (left, top, width, height)
# and crop the image using them
def cropImageUsingBounds(image, coordinates):
    # cropCoordinates is in format left, upper, right, lower
    # which pillow's crop wants
    cropCoordinatesPillow = (coordinates.x, coordinates.y,
                             coordinates.x + coordinates.w,
                             coordinates.y + coordinates.h)

    return image.crop(cropCoordinatesPillow)


outputs = []
i = 0
numPoints = str(len(dset))
numSkipped = 0
numClass0Right = 0
numClass0Wrong = 0
numClass1Right = 0
numClass1Wrong = 0

def printCurrentStats():
    print("percent right: " + str((numClass0Right + numClass1Right)/i))
    print("percent wrong: " + str((numClass0Wrong + numClass1Wrong)/i))
    print("percent skipped: " + str(numSkipped / i))
    print("class 0 right: " + str(numClass0Right))
    print("class 0 wrong: " + str(numClass0Wrong))
    print("class 1 right: " + str(numClass1Right))
    print("class 1 wrong: " + str(numClass1Wrong))
    print("skipped: " + str(numSkipped))

os.system("mkdir -p " + output_dir_class0 + "/right/")
os.system("mkdir -p " + output_dir_class0 + "/wrong/")
os.system("mkdir -p " + output_dir_class0 + "/heatmap/")
os.system("mkdir -p " + output_dir_class1 + "/right/")
os.system("mkdir -p " + output_dir_class1 + "/wrong/")
os.system("mkdir -p " + output_dir_class1 + "/heatmap/")
os.system("mkdir -p " + output_dir_skip)
for dataPoint in dset:
    pil_image_unresized, labelIndex = dataPoint
    pil_image = pil_image_unresized.resize((imageMaxScale, imageMaxScale))
    print("Working on element " + str(i) + " of " + numPoints, flush=True)
    fileName = os.path.basename(dset.imgs[i][0])
    fileFullPath = dset.imgs[i][0]
    fileNameWithoutExt = os.path.splitext(fileName)[0]
    i += 1

    imageRegions = []
    imageRegionsAsTensors = []
    for regionCoordinate in regionCoordinates:
        croppedImage = cropImageUsingBounds(pil_image, regionCoordinate)
        imageRegions.append(croppedImage)
        imageAsTensorForEval = data_transforms(croppedImage)
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
    # in mostLikely, first [0] gives the probabilities of the boxes that are most likely to be in each class. skip if both max probabilities < 0.9
    # second [0] gives the probability of the element with the max probability
    # of being the first class (0 indexing)
    print("max probabilites are " + str(mostLikely[0]))
    if mostLikely[0][0] < categoryThreshold and mostLikely[0][1] < categoryThreshold:
        print("dropping image " + fileName + " as probabilites were all less than " + str(categoryThreshold))
        #makeAndSaveToFileCamClassificationHeatmap(model_input_location, fileFullPath,
        #                                          output_dir_skip + "/" + fileNameWithoutExt + "_0.jpg", label_map, 0)
        #makeAndSaveToFileCamClassificationHeatmap(model_input_location, fileFullPath,
        #                                          output_dir_skip + "/" + fileNameWithoutExt + "_1.jpg", label_map, 1)
        numSkipped += 1
    # write to the folder for class 0 or 1 depending on which is most likely
    # if likely to be in both classes, write to both
    if mostLikely[0][0] > categoryThreshold:
        print("think image " + fileName + " is class 0 as most likely object was: " + str(mostLikely[0]))
        indexOfMostLikely = classProbabilityTensor.max(0)[1][0]
        # [1] gives the indices instead of the probabilities
        pil_image.save(output_dir_class0 + "/" + fileName)
        #make the cam heatmap for this class
        #makeAndSaveToFileCamClassificationHeatmap(model_input_location, output_dir_class0 + "/" + fileName,
        #                                 output_dir_class0 + "/heatmap/" + fileName, label_map, 0)
        if labelIndex == 0:
            numClass0Right += 1
            imageRegions[indexOfMostLikely].save(output_dir_class0 + "/right/" + fileName)
        else:
            numClass0Wrong += 1
            imageRegions[indexOfMostLikely].save(output_dir_class0 + "/wrong/" + fileName)
    if mostLikely[0][1] > categoryThreshold:
        print("think image " + fileName + " is class 1 as most likely object was: " + str(mostLikely[0]))
        indexOfMostLikely = classProbabilityTensor.max(0)[1][1]
        # [1] gives the indices instead of the probabilities
        pil_image.save(output_dir_class1 + "/" + fileName)
        #makeAndSaveToFileCamClassificationHeatmap(model_input_location, output_dir_class1 + "/" + fileName,
        #                                 output_dir_class1 + "/heatmap/" + fileName, label_map, 1)
        if labelIndex == 1:
            numClass1Right += 1
            imageRegions[indexOfMostLikely].save(output_dir_class1 + "/right/" + fileName)
        else:
            numClass1Wrong += 1
            imageRegions[indexOfMostLikely].save(output_dir_class1 + "/wrong/" + fileName)
    printCurrentStats()