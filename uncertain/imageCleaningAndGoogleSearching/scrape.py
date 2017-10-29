from splinter import Browser
import subprocess
from random import randint
import time
from os.path import isfile, join
import os
import sys

# first input is input directory to read from
# second input is output directory
# third, optional input is a string of inputs separated by spaces to filter scraped images based on
dirToDownload = sys.argv[1]
outputDir = sys.argv[2]
if len(sys.argv) == 4:
    scrapeKeywords = sys.argv[3].lower().split(" ")
    print("scrapeKeywords are:")
    for keyword in scrapeKeywords:
        print(keyword)
else:
    scrapeKeywords = False
scriptDir = os.path.dirname(os.path.realpath(sys.argv[0]))

imagesToDownloadPerImage=100

# from https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
imageFiles = [f for f in os.listdir(dirToDownload) if isfile(join(dirToDownload, f))]

def setTimeoutTime(browser, timeoutSeconds):
    browser.driver.set_page_load_timeout(timeoutSeconds)

with Browser() as browser:
    # only wait on pages for 10 seconds, not 15 minutes
    for imgFileName in imageFiles:
        # google not going to timeout, want to wait for it to return
        setTimeoutTime(browser, 300)
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

        # don't trust the websites google is sending me too, timeout quickly on them
        setTimeoutTime(browser, 10)
        for image in images:
            if numDownloaded > imagesToDownloadPerImage:
                break
            numDownloaded += 1
            try:
                browser.visit(image)
                time.sleep(0.1)
                if len(browser.find_by_css(".irc_fsl.irc_but.i3596")) == 0:
                    print("skipping image " + str(numDownloaded - 1) + " as no View Image button", flush=True)
                    print(browser.url, flush=True)
                    continue
                elif scrapeKeywords != False:
                    noMatchingKeywords = True
                    pageHTML = browser.html.lower()
                    for keyword in scrapeKeywords:
                        if pageHTML.find(keyword) > -1:
                            noMatchingKeywords = False
                            matchKeyword = keyword
                            break
                    if noMatchingKeywords:
                        print("skipping image " + str(numDownloaded - 1) + " as matches no keywords in description", flush=True)
                        print(browser.url, flush=True)
                        continue
                    else:
                        print("downloading image " + str(numDownloaded - 1) + " that matches keyword " + matchKeyword, flush=True)
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
                print("Browser url for exception", flush=True)
                print(browser.url, flush=True)


        for process in processList:
            process.wait()

        #end_time = time.time()
        #if end_time > start_time + wait_time:
        #    time.sleep(end_time - start_time - wait_time)
