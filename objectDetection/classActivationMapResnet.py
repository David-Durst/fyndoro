# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# from https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py

import sys
import json
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
import cv2

# if this is empty, the default model will be used
model_input_location = sys.argv[1]
input_image = sys.argv[2]
label_map = sys.argv[3]
desired_label_index = int(sys.argv[4])
output_location = sys.argv[4]


net = models.resnet18(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
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

img_pil = Image.open(input_image)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)

# download the imagenet category list
classes = {int(key):value for (key, value)
          in json.load(open(label_map)).items()}

h_x = F.softmax(logit).data.squeeze()
probs, idx = h_x.sort(0, True)

# output the prediction
for i in range(0, 2):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [desired_label_index])

# render the CAM and output
print('showing heatmap for label %s'%classes[desired_label_index])
img = cv2.imread(input_image)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)