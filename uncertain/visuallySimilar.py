import io
import os
from os import listdir
from os.path import isfile, join, splitext
import urllib.request


# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
dirsToDownload = ['images/1.0,0.0', 'images/0.85,0.15', 'images/0.7,0.3']

for dir in dirsToDownload:
    if not os.path.exists(dir):
        os.makedirs(dir)

numFilesDownloaded = 0
# go through all directories and download next layer of images
for i in range(len(dirsToDownload) - 1):
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

# go through all directories, remove not jpg or png and resize the jpg and png
for dir in dirsToDownload:
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
