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
        # go from url reverse image to file upload image
        #browser.find_by_text("Upload an image").click()
        # upload a file
        browser.attach_file("encoded_image", join(dirToDownload, imgFileName))
        #print("start")
        #print(dirToDownload)
        #print(imgFileName)
        #print("b1")
        #print(browser.html)
        #print("b2")
        visSimLink = browser.find_by_text("Visually similar images")
        print(browser.url)
        #print(visSimLink)
        #print(browser.html)
        if len(visSimLink) == 0:
            continue
        visSimLink.click()
        images = [x['href'] for x in browser.find_by_css(".rg_l")]
        #images = browser.find_by_css(".rg_ic.rg_i")
        numDownloaded = 0

        print("images to download " + str(len(images)))
        print(imgFileName)
        processList = []

        for image in images:
            if numDownloaded > imagesToDownloadPerImage:
                break
            #image.click()
            #browser.find_by_css(".irc_fsl.irc_but.i3596")[1].click()
            #time.sleep(0.5)
            #imgToDownload = browser.windows[1].url
            #print(imgToDownload)
            #time.sleep(0.5)
            #imgToDownload = browser.windows[1].url
            browser.visit(image)
            #print(browser.url)
            if len(browser.find_by_css(".irc_fsl.irc_but.i3596")) == 0:
                continue
            imgToDownload = browser.find_by_css(".irc_fsl.irc_but.i3596")[1]['href']
            #print("a")
            #try:
            #    browser.visit(browser.find_by_css(".irc_fsl.irc_but.i3596")[1]['href'])
            #except:
            #    pass
            #imgToDownload = browser.windows[0].url
            #print(imgToDownload)
            #process = subprocess.Popen("wget -t 3 --directory-prefix " + outputDir + " " + imgToDownload, shell=True)
            print("timeout 15 python " + scriptDir + "/scrapeDLSubprocess.py " + imgToDownload + " " + outputDir)
            process = subprocess.Popen("timeout 15 python " + scriptDir + "/scrapeDLSubprocess.py " + imgToDownload + " " + outputDir, shell=True)
            #browser.windows[1].close()
            processList.append(process)
            numDownloaded += 1


        for process in processList:
            process.wait()

        #end_time = time.time()
        #if end_time > start_time + wait_time:
        #    time.sleep(end_time - start_time - wait_time)

