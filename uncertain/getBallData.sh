#!/usr/bin/env bash
set -x
# note, assume already have imagenet_balls and all images from datasets extracted into an all subdirectory for each type
scriptDir=$(dirname "$(greadlink -f "$0")")
images=$scriptDir/imagenet_balls
categoryGroups=(tennis_balls baseballs)
googleCat=tennis_balls
valImages=700
trainImages=200
trainIncrements=25
numIncrements=$(expr $trainImages / $trainIncrements)
valIncrements=$(expr $valImages / $numIncrements)

# get all the training and validation data cleaned and split up
for c in "${categoryGroups[@]}"
do
    catImages=$images/$c
    rm -rf $catImages/train
    rm -rf $catImages/val
    mkdir -p $catImages/train
    mkdir -p $catImages/val
    # put all images for use in train first, then mv a subset of them to val
    gshuf -n $(expr $valImages + $trainImages) -e $catImages/all/* | xargs -I {} cp {} $catImages/train/
    python imageCleaningAndGoogleSearching/clean.py $catImages/train
    gshuf -n $valImages -e $catImages/train/* | xargs -I {} mv {} $catImages/val/

    # split the datasets into 25 image groups
    for i in $(seq $numIncrements)
    do
        subsetCatImages=$catImages/subset_${i}_of_${trainIncrements}_training_images
        rm -rf $subsetCatImages
        mkdir -p $subsetCatImages
        mkdir -p $subsetCatImages/noGoogleTrain
        mkdir -p $subsetCatImages/train
        mkdir -p $subsetCatImages/val

        gshuf -n $trainIncrements -e $catImages/train/* | xargs -I {} mv {} $subsetCatImages/noGoogleTrain/
        cp $subsetCatImages/noGoogleTrain/* $subsetCatImages/train/
        gshuf -n $valIncrements -e $catImages/val/* | xargs -I {} mv {} $subsetCatImages/val/
        exit
        python $scriptDir/imageCleaningAndGoogleSearching/search.py $subsetCatImages/noGoogleTrain $subsetCatImages/train
        python $scriptDir/imageCleaningAndGoogleSearching/clean.py $subsetCatImages/train

        # join this with previous subsets to create training and validation runs of increasing size
        # this means that subset 1 leads to a merged subset of 25, subset 2 joins with subset 1 to make a merged subset of 50
        mergedSubset=$catImages/merged_$(expr $trainIncrements * $i)_training_images
        mkdir -p $mergedSubset
        cp -r $subsetCatImages $mergedSubset
        if [ $i -gt 1 ]
        then
            cp -RT $previousMergedSubset $mergedSubset
        fi
        previousMergedSubset=mergedSubset
    done
done





