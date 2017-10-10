#!/usr/bin/env bash
# this file is only to be run on the scraping machine
scriptDir=$(dirname "$(readlink -f "$0")")
# first argument is the location of the folder with the images.
# this folder should have two subfolders, 1.0,0.0 with all the images
# in the first class from imagenet, and 0.0,1.0 for the images in the
# second class
imageDirectory=$1
numTrials=3
for i in $(seq $numTrials)
do
    trialLocation=$scriptDir/${imageDirectory}_trial$i
    mkdir $trialLocation
    ln -sr $scriptDir/${imageDirectory}/1.0,0.0 $trialLocation/1.0,0.0
    ln -sr $scriptDir/${imageDirectory}/0.0,1.0 $trialLocation/0.0,1.0
    ${scriptDir}/makeDataForExperiments.sh $scriptDir/${imageDirectory}_trial$i
done