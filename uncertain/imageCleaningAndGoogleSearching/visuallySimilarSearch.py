# Imports the Google Cloud client library
import os
import io
from os.path import isfile, join, splitext
import json
import sys
import subprocess
import base64
import requests

# first input is input directory to read from
# second input is output directory
# third input is the location of the file containing the google api key
dirToDownload = sys.argv[1]
outputDir = sys.argv[2]
keyFile = sys.argv[3]
with open(keyFile, 'r') as f:
    apiKey = f.read()

apiUrl = "https://vision.googleapis.com/v1/images:annotate?key={}".format(apiKey)
requestJSONTemplate = {
  "requests": [
    {
      "image": {"content": "toBeReplaced"}, "features": [{"type": "WEB_DETECTION"}]
    }
]}

# from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
imageFiles = [f for f in os.listdir(dirToDownload) if isfile(join(dirToDownload, f))]

processList = []

# go through each file in the directory and download all partial matches
for imgFileName in imageFiles:
    # Loads the image into memory
    with io.open(join(dirToDownload, imgFileName), 'rb') as image_file:
        content = image_file.read()

    requestJSONTemplate['requests'][0]['image']['content'] = base64.b64encode(content).decode("utf-8")
    response = requests.post(url=apiUrl, data=json.dumps(requestJSONTemplate))
    responseJson = json.loads(response.text)

    for imgToDownload in responseJson['responses'][0]['webDetection']['visuallySimilarImages']:
        process = subprocess.Popen("wget --quiet -t 3 --directory-prefix " + outputDir + " " + imgToDownload['url'], shell=True)
        processList.append(process)

for process in processList:
    process.wait()
