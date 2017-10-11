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

imagesToDownloadPerImage=100

# from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
imageFiles = [f for f in os.listdir(dirToDownload) if isfile(join(dirToDownload, f))]


with Browser() as browser:
    # only wait on pages for 10 seconds, not 15 minutes
    browser.driver.set_page_load_timeout(10)
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
        while browser.find_by_id("qbupm"):
            print("stalling for image", flush=True)
            time.sleep(0.5)
        visSimLink = browser.find_by_text("Visually similar images")
        if len(visSimLink) == 0:
            print("didn't find vissim", flush=True)
            print(browser.url, flush=True)
            print(browser.html, flush=True)
            continue
        visSimLink.click()
        time.sleep(0.5)
        images = [x['href'] for x in browser.find_by_css(".rg_l")]
        numDownloaded = 0

        print("images to download " + str(len(images)), flush=True)
        print(imgFileName, flush=True)
        processList = []

        for image in images:
            if numDownloaded > imagesToDownloadPerImage:
                break
            numDownloaded += 1
            try:
                browser.visit(image)
                time.sleep(0.1)
                if len(browser.find_by_css(".irc_fsl.irc_but.i3596")) == 0:
                    print("skipping image " + str(numDownloaded - 1), flush=True)
                    print(browser.html, flush=True)
                    continue
                else:
                    print("downloading image " + str(numDownloaded - 1), flush=True)
                imgToDownload = browser.find_by_css(".irc_fsl.irc_but.i3596")[1]['href']
                if imgToDownload is None:
                    print("website does redirect, imgToDownload is None")
                    continue
                print("timeout 30 wget -t 3 --directory-prefix " + outputDir + " \"" + imgToDownload + "\"", flush=True)
                process = subprocess.Popen("timeout 30 wget -t 3 --directory-prefix " + outputDir + " \"" + imgToDownload + "\"", shell=True)
                processList.append(process)
                #process.wait()
            except Exception as e:
                print("Caught exception", flush=True)
                print(e, flush=True)
                try:
                    browser.get_alert().dismiss()
                except Exception:
                    print("Exception not alert", flush=True)
                print("Browser data", flush=True)
                print(browser.url, flush=True)


        for process in processList:
            process.wait()

        #end_time = time.time()
        #if end_time > start_time + wait_time:
        #    time.sleep(end_time - start_time - wait_time)
