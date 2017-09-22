# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types
import os
import io
from os.path import isfile, join, splitext
import sys
import subprocess

# first input is input directory to read from
# second input is output directory
dirToDownload = sys.argv[1]
outputDir = sys.argv[2]

# Instantiates a client
client = vision.ImageAnnotatorClient()

# from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
imageFiles = [f for f in os.listdir(dirToDownload) if isfile(join(dirToDownload, f))]

processList = []

# go through each file in the directory and download all partial matches
for imgFileName in imageFiles:
    # Loads the image into memory
    with io.open(join(dirToDownload, imgFileName), 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.web_detection(image=image)
    notes = response.web_detection

    for imgToDownload in notes.partial_matching_images:
        process = subprocess.Popen("wget -t 3 --directory-prefix " + outputDir + " " + imgToDownload.url, shell=True)
        processList.append(process)

for process in processList:
    process.wait()
