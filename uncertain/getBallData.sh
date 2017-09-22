#!/usr/bin/env bash
set -x
# note, assume already have imagenet_balls and all images from datasets extracted into an all subdirectory for each type
scriptDir=$(dirname "$(greadlink -f "$0")")
images=$scriptDir/imagenet_balls
categoryGroups=("1.0,0.0" "0.0,1.0")
googleCat="1.0,0.0"
googleCatOutput="0.8,0.2"
valImages=700
trainImages=200
trainIncrements=25
numIncrements=$(expr $trainImages / $trainIncrements)
valIncrements=$(expr $valImages / $numIncrements)

# get all the training and validation data cleaned and split up
for c in "${categoryGroups[@]}"
do
    rm -rf $images/train/$c
    rm -rf $images/val/$c
    mkdir -p $images/train/$c
    mkdir -p $images/val/$c
    # put all images for use in train first, then mv a subset of them to val
    gshuf -n $(expr $valImages + $trainImages) -e $images/$c/* | xargs -I {} cp {} $images/train/$c/
    python imageCleaningAndGoogleSearching/clean.py $images/train/$c
    gshuf -n $valImages -e $images/train/$c/* | xargs -I {} mv {} $images/val/$c/
done

exit
# split the datasets into 25 image groups
for i in $(seq $numIncrements)
do
    subsetImages=$images/subset_${i}_of_${trainIncrements}_training_images
    rm -rf $subsetImages
    mkdir -p $subsetImages
    mkdir -p $subsetImages/train
    mkdir -p $subsetImages/val
    for c in "${categoryGroups[@]}"
    do
        gshuf -n $trainIncrements -e $images/train/$c/* | xargs -I {} mv {} $subsetImages/train/$c/
        gshuf -n $valIncrements -e $images/val/$c/* | xargs -I {} mv {} $subsetImages/val/$c/
        if [ $c -eq $googleCat ]
        then
            python $scriptDir/imageCleaningAndGoogleSearching/search.py $subsetImages/train/$c/ $subsetImages/train/$googleCatOutput/
            python $scriptDir/imageCleaningAndGoogleSearching/clean.py $subsetImages/train/$googleCatOutput/
        fi
    done

    # join this with previous subsets to create training and validation runs of increasing size
    # this means that subset 1 leads to a merged subset of 25, subset 2 joins with subset 1 to make a merged subset of 50
    mergedSubset=$images/merged_$(expr $trainIncrements * $i)_training_images
    mkdir -p $mergedSubset
    cp -r $subsetCatImages $mergedSubset
    if [ $i -gt 1 ]
    then
        cp -RT $previousMergedSubset $mergedSubset
    fi
    previousMergedSubset=mergedSubset
done