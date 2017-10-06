from splinter import Browser
import subprocess
from random import randint
import time
from os.path import isfile, join
import os
import sys

# first input is input directory to read from
# second input is output directory
dirToDownload = sys.argv[1]
outputDir = sys.argv[2]
scriptDir = os.path.dirname(os.path.realpath(sys.argv[0]))

imagesToDownloadPerImage=30

# from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
imageFiles = [f for f in os.listdir(dirToDownload) if isfile(join(dirToDownload, f))]


with Browser() as browser:
    for imgFileName in imageFiles:
        browser.visit("https://images.google.com/")
        #start_time = time.time()
        # wait a random number of seconds between 3 and 5 to ensure don't get blocked
        # by google for being a bot
        #wait_time = randint(4,6)
        # get the reverse image section open
        browser.find_by_id("qbi").click()
        # upload a file
        browser.attach_file("encoded_image", join(dirToDownload, imgFileName))
        time.sleep(0.5)
        visSimLink = browser.find_by_text("Visually similar images")
        if len(visSimLink) == 0:
            continue
        visSimLink.click()
        time.sleep(0.5)
        images = [x['href'] for x in browser.find_by_css(".rg_l")]
        numDownloaded = 0

        print("images to download " + str(len(images)))
        print(imgFileName)
        processList = []

        for image in images:
            if numDownloaded > imagesToDownloadPerImage:
                break
            browser.visit(image)
            time.sleep(0.1)
            if len(browser.find_by_css(".irc_fsl.irc_but.i3596")) == 0:
                continue
            imgToDownload = browser.find_by_css(".irc_fsl.irc_but.i3596")[1]['href']
            print("timeout 15 python \"" + scriptDir + "/scrapeDLSubprocess.py\" \"" + imgToDownload + "\" " + outputDir)
            process = subprocess.Popen("timeout 30 python \"" + scriptDir + "/scrapeDLSubprocess.py\" \"" + imgToDownload + "\" " + outputDir, shell=True)
            #processList.append(process)
            process.wait()
            numDownloaded += 1


        for process in processList:
            process.wait()

        #end_time = time.time()
        #if end_time > start_time + wait_time:
        #    time.sleep(end_time - start_time - wait_time)
