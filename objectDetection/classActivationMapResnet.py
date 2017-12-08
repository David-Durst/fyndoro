# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# from https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py

import sys
import json
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import os
import torch
import numpy as np
import cv2


def generateCamClassificationHeatmap(model_input_location, input_image, label_map, desired_label_index):
    net = models.resnet101(pretrained=True)
    #for param in model.parameters():
    #    param.requires_grad = False
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)
    # if model_input_location is empty, use default weights
    if model_input_location != "":
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        # no idea what the second and third parameters to torch.load do,
        # but they fix issue of loading model trained on gpu on a host with only cpu, from the link above
        net.load_state_dict(torch.load(model_input_location, map_location=lambda storage, loc: storage))

    finalconv_name = 'layer4'
    net.eval()

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(finalconv_name).register_forward_hook(hook_feature)

    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())

    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam


    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Scale((224,224)),
       transforms.ToTensor(),
       normalize
    ])

    img_pil = Image.fromarray(cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB))
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)

    # download the imagenet category list
    classes = {int(key):value for (key, value)
              in json.load(open(label_map)).items()}

    h_x = F.softmax(logit).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output the prediction
    for i in range(0, 3):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [desired_label_index])

    # render the CAM and output
    print('showing heatmap for label %s'%classes[desired_label_index])
    height, width, _ = input_image.shape
    grayscaleHeatmap = cv2.resize(CAMs[0],(width, height))
    heatmap = cv2.applyColorMap(grayscaleHeatmap, cv2.COLORMAP_JET)
    return (grayscaleHeatmap, input_image, heatmap * 0.3 + input_image * 0.5)

def getConnectedComponentsAndImgData(model_input_location, input_image, label_map, desired_label_index):
    grayscaleHeatmap, img, imgAndColorHeatmap = generateCamClassificationHeatmap(model_input_location, input_image, label_map,
                                                                     desired_label_index)
    # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    # not that cv2.THRESH_BINARY and cv2.THRESH_OTSU are flags, binary says binary thresholding (i think)
    # otsu automatically figures out the best global thresholding
    # https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html
    ret, thresh = cv2.threshold(grayscaleHeatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 4
    # invert with bitwise_not as thresh makes the desired regions black and the rest white, but
    # connected components finds the white regions
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    num_labels, labels, stats, centroids = output
    # sort stats so one with largest area comes first
    # fifth element of each element in stats is size, so getting largest of those
    statsSorted = stats[np.argsort(stats[:, 4])[::-1]]
    return (statsSorted, thresh, grayscaleHeatmap, img, imgAndColorHeatmap)

def getLargestConnectComponentAsPILImage(model_input_location, input_image, label_map, desired_label_index):
    statsSorted, thresh, grayscaleHeatmap, img, _ = getConnectedComponentsAndImgData(model_input_location, input_image, label_map, desired_label_index)
    # filter out regions that aren't greater than average by at least 20
    # meaning at some significant part of region above threshold
    meanPixelValue = np.mean(thresh)
    # reverse index order here as array index is row then column, aka y then x
    aboveThresholdStats = [s for s in statsSorted if np.mean(thresh[s[1]:(s[1]+s[3]), s[0]:(s[0] + s[2])]) > meanPixelValue]
    # give up if good regions
    if len(aboveThresholdStats) == 0:
        return None
    imgsToReturn = []
    # already sorted, so 0 gets largest
    for regionStat in aboveThresholdStats:
        # object is of form leftmost x, topmost y, wigth, height, size
        x, y, width, height, size = regionStat
        # note that 0,0 is top left in opencv
        # taking subset of image in bounding box
        connectedComponentImg = img[x:(x + width), y:(y+height)]
        # https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imread
        # that shows that default color scheme is BGR, not RGB
        # https://stackoverflow.com/questions/13576161/convert-opencv-image-into-pil-image-in-python-for-use-with-zbar-library
        # that provides how to do conversion
        if cv2.cvtColor(connectedComponentImg,cv2.COLOR_BGR2RGB) is None:
            print("skipping a region")
            continue
        imgsToReturn.append(Image.fromarray(cv2.cvtColor(connectedComponentImg,cv2.COLOR_BGR2RGB)))
    if len(imgsToReturn) == 0:
        return None
    return imgsToReturn


def makeAndSaveToFileCamClassificationHeatmap(model_input_location, input_image_location, output_image_location, label_map, desired_label_index):
    input_image = cv2.imread(input_image_location)
    statsSorted, thresh, grayscaleHeatmap, img, imgAndColorHeatmap = getConnectedComponentsAndImgData(model_input_location, input_image,
                                                                                     label_map, desired_label_index)
    meanPixelValue = np.mean(thresh)
    print("full image shape: " + str(img.shape))
    print("Mean pixel value: " + str(meanPixelValue))
    print("Stats sorted: " + str(statsSorted))
    for regionStat in statsSorted:
        print("region " + str(regionStat) + " mean pixel value " + str(
            np.mean(thresh[regionStat[1]:(regionStat[1] + regionStat[3]), regionStat[0]:(regionStat[0] + regionStat[2])])))
        cv2.rectangle(imgAndColorHeatmap, (regionStat[0], regionStat[1]),
                      (regionStat[0] + regionStat[2], regionStat[1] + regionStat[3]), (255, 0, 0), 10)
    cv2.imwrite(output_image_location, imgAndColorHeatmap)
    #cv2.imwrite(os.path.dirname(output_image_location) + "/grayscale.jpg", grayscaleHeatmap)
    #cv2.imwrite(os.path.dirname(output_image_location) + "/thresh.jpg", thresh)


if __name__ == "__main__":
    makeAndSaveToFileCamClassificationHeatmap(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))