from splinter import Browser
import subprocess
from random import randint
import time
from os.path import isfile, join
import os
import sys

# first input is the text to search
# second input is output directory
# third, optional input is a string of inputs separated by spaces to filter scraped images based on
textToSearch = sys.argv[1]
outputDir = sys.argv[2]
if len(sys.argv) == 5:
    scrapeKeywords = sys.argv[3].lower().split(" ")
    scrapeWrongwords = sys.argv[4].lower().split(" ")
    print("scrapeKeywords are:")
    for keyword in scrapeKeywords:
        print(keyword)
else:
    scrapeKeywords = False
scriptDir = os.path.dirname(os.path.realpath(sys.argv[0]))

imagesToDownloadPerImage=100

def setTimeoutTime(browser, timeoutSeconds):
    browser.driver.set_page_load_timeout(timeoutSeconds)

with Browser() as browser:
    # google not going to timeout, want to wait for it to return
    setTimeoutTime(browser, 300)
    browser.visit("https://images.google.com/")
    # enter the search text in the search box
    browser.type("q", textToSearch)
    browser.click_link_by_id("_fZl")
    time.sleep(0.5)

    images = [x['href'] for x in browser.find_by_css(".rg_l")]
    numDownloaded = 0

    print("images to download " + str(len(images)), flush=True)
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
                pageHTML = browser.html.lower()
                if not all(keyword in pageHTML for keyword in scrapeKeywords):
                    print("skipping image " + str(numDownloaded - 1) + " as matches not all keywords in description",
                          flush=True)
                    print(browser.url, flush=True)
                    continue
                elif any(wrongWord in pageHTML for wrongWord in scrapeWrongwords):
                    print("skipping image " + str(numDownloaded - 1) + " as matches wrongwords in description", flush=True)
                    print(browser.url, flush=True)
                    continue
                else:
                    print("downloading image " + str(numDownloaded - 1) + " that matches all keywords, no wrongwords", flush=True)
            else:
                print("downloading image " + str(numDownloaded - 1), flush=True)
            imgToDownload = browser.find_by_css(".irc_fsl.irc_but.i3596")[1]['href']
            if imgToDownload is None:
                print("website does redirect, imgToDownload is None")
                continue
            print("timeout 30 wget -t 3 --directory-prefix " + outputDir + " \"" + imgToDownload + "\"", flush=True)
            process = subprocess.Popen("timeout 30 wget -t 3 --directory-prefix " + outputDir + " \"" + imgToDownload + "\"", shell=True)
            processList.append(process)
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