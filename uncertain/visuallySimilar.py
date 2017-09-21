import os
from os import listdir
from os.path import isfile, join, splitext

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

def makeDirs(dirsArr):
    for dir in dirsArr:
        if not os.path.exists(dir):
            os.makedirs(dir)

imagesFolder = "images/"
# the sizes of the training data (from imagenet superset, not google) and
# the validation sets in terms of number of images
# this will be the top level directory for the image folder (under images)
numData = ["50"]


# The name of the image file to annotate
dirsToDownload = ['images/1.0,0.0', 'images/0.85,0.15', 'images/0.7,0.3']

makeDirs(dirsToDownload)

# go through all directories and download next layer of images
for i in range(len(dirsToDownload) - 1):
    break
    # from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    imageFiles = [f for f in listdir(dirsToDownload[i]) if isfile(join(dirsToDownload[i], f))]

    # go through each file in the directory and download all partial matches
    for imgFileName in imageFiles:
        # Loads the image into memory
        with io.open(join(dirsToDownload[i], imgFileName), 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        response = client.web_detection(image=image)
        notes = response.web_detection

        for imgToDownload in notes.partial_matching_images:
            os.system("wget --directory-prefix " + dirsToDownload[i+1] + " " + imgToDownload.url)


# download, extract list of images URLs for parfaits, which are good data as not in
# imagenet 1000 dataset this was trained on
if not os.path.exists("pairfait_images/parfait_img_urls_randomized.txt"):
    os.system("wget --directory-prefix parfait_images http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07616386")
    os.system("gshuf imagenet.synset.geturls\?wnid\=n07616386 > parfait_img_urls_randomized.txt")

# ran wc -l, 1230 images in file, make folders of every 25
with open("parfait_img_urls_randomized.txt") as f:
    for line in f:
        break
        os.system("wget --tries=1 --directory-prefix images/0.0,1.0 " + line.split("\t")[1])

dirsToFormat = dirsToDownload + ["images/0.0,1.0"]
# go through all directories, remove not jpg or png and resize the jpg and png
for dir in dirsToFormat:
    anyFiles = [f for f in listdir(dir) if isfile(join(dir, f))]

    for f in anyFiles:
        _, fExt = splitext(f)

        relPathAndName = join(dir, f)

        # delete non images (and duplicates, which have a .1 or .n at end, where n is an integer)
        if fExt != ".png" and fExt != ".jpg":
            os.remove(relPathAndName)
        # otherwise, reshape them
        else:
            os.system("convert -resize 500x500 \'" + relPathAndName + "\' \'" + relPathAndName + "\'")
