from splinter import Browser
import subprocess
import sys
import os

imageToDownloadPage = sys.argv[1]
outputDir = sys.argv[2]

with Browser() as browser:
    browser.visit(imageToDownloadPage)
    imgToDownload = browser.windows[0].url
    os.system("wget -t 3 --directory-prefix " + outputDir + " " + imgToDownload, shell=True)
