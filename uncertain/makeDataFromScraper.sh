#!/usr/bin/env bash
# this file is only to be run on the scraping machine
scriptDir=$(dirname "$(readlink -f "$0")")
# first argument is the location of the folder with the images.
# this folder should have two subfolders, 1.0,0.0 with all the images
# in the first class from imagenet, and 0.0,1.0 for the images in the
# second class
imageDirectory=$1
# second argument is where to rsync the data to, like durst@dawn4:path/to/put/in
# path will have trial number appended to it
destinationServer=$2
numTrials=3
for i in $(seq $numTrials)
do
    cp -r $scriptDir/$imageDirectory $scriptDir/${imageDirectory}_trial$i
    ${scriptDir}/makeDataForExperiments.sh $scriptDir/${imageDirectory}_trial$i
done