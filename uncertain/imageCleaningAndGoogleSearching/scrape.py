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

imagesToDownloadPerImage=30

# from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
imageFiles = [f for f in os.listdir(dirToDownload) if isfile(join(dirToDownload, f))]


with Browser() as browser:
    for imgFileName in imageFiles:
        browser.visit("https://images.google.com/")
        start_time = time.time()
        # wait a random number of seconds between 3 and 5 to ensure don't get blocked
        # by google for being a bot
        wait_time = randint(3,5)
        # get the reverse image section open
        browser.find_by_id("qbi").click()
        # go from url reverse image to file upload image
        #browser.find_by_text("Upload an image").click()
        # upload a file
        browser.attach_file("encoded_image", join(dirToDownload, imgFileName))
        browser.find_by_text("Visually similar images").click()
        images = browser.find_by_css(".rg_ic.rg_i")
        numDownloaded = 0

        processList = []

        for image in images:
            if numDownloaded > imagesToDownloadPerImage:
                break
            image.click()
            browser.find_by_css(".irc_fsl.irc_but.i3596")[1].click()
            process = subprocess.Popen("wget -t 3 --directory-prefix " + outputDir + " " + imgToDownload.url, shell=True)
            processList.append(process)
            numDownloaded += 1


        for process in processList:
            process.wait()

        end_time = time.time()
        if end_time > start_time + wait_time:
            time.sleep(end_time - start_time - wait_time)
