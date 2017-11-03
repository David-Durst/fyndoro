from splinter import Browser
import subprocess
from random import randint
import time
from os.path import isfile, join
import os
import sys

# first input is input directory to read from
# second input is output directory
queryToDownload = sys.argv[1]
outputDir = sys.argv[2]
scriptDir = os.path.dirname(os.path.realpath(sys.argv[0]))

# don't download more than 200 as those are likely ads, have weird
# redirects
imagesToDownload = 200

#"diet pepsi" vs "diet dr pepper"
#python imageCleaningAndGoogleSearching/scrapeEbay.py "diet pepsi" /home/durst/fyndoro/uncertain/userdatasets/drpepperOrPepsi/val0.0,1.0 &> userdatasets/drpp.log
#python imageCleaningAndGoogleSearching/scrapeEbay.py "diet dr pepper" /home/durst/fyndoro/uncertain/userdatasets/drpepperOrPepsi/val1.0,0.0 &> userdatasets/drppOther.log
with Browser() as browser:
    # only wait on pages for 10 seconds, not 15 minutes
    browser.driver.set_page_load_timeout(10)
    browser.visit("https://www.ebay.com/")
    #start_time = time.time()
    # search for the query
    browser.find_by_css("#gh-ac").fill(queryToDownload)
    browser.find_by_css("#gh-btn").click()
    # get 200 results if possible
    # this checks if the dropdown down toggle exists, if more than 50 results
    if len(browser.find_by_css(".dropdown-toggle")) >= 3:
        # this toggles the drop down menu
        browser.find_by_css(".dropdown-toggle")[2].click()
        # this picks the menu item for
        browser.find_by_css(".ipp")[3].click()
    time.sleep(0.5)
    listingPages = [x['href'] for x in browser.find_by_css(".vip")]
    numDownloaded = 0

    processList = []

    for listing in listingPages:
        if numDownloaded >= imagesToDownload:
            break
        numDownloaded += 1
        try:
            browser.visit(listing)
            browser.find_by_id("linkMainImg").click()
            browser.find_by_id("vi_zoom_trigger_mask").click()
            time.sleep(0.2)
            print("downloading image for " + listing, flush=True)
            imgToDownload = browser.find_by_id("viEnlargeImgLayer_img_ctr")['src']
            processStr = "timeout 30 wget -t 3 -O " + outputDir + "/" + str(numDownloaded) + ".jpg \"" + imgToDownload + "\""
            print(processStr, flush=True)
            process = subprocess.Popen(processStr, shell=True)
            processList.append(process)
            #process.wait()
        except Exception as e:
            print("Caught exception", flush=True)
            print(e, flush=True)
            print("page was " + listing, flush=True)
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
