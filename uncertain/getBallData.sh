#!/usr/bin/env bash
# note, assume already have imagenet_balls and all images from datasets extracted into an all subdirectory for each type
images=imagenet_balls
categoryGroups=(tennis_balls baseballs)
googleCat=tennis_balls
valImages=700
trainImages=200
trainIncrements=25
numIncrements=$(expr trainImages / trainIncrements)
valIncrements=$(expr valImages / numIncrements)

# get all the training and validation data cleaned and split up
for c in "${categoryGroups[@]}"
do
    catImages=$images/$c
    rm -rf $catImages/train
    rm -rf $catImages/val
    mkdir -p $catImages/train
    mkdir -p $catImages/val
    # put all images for use in train first, then mv a subset of them to val
    gshuf -n $(expr valImages + trainImages) -e $catImages/all/* | xargs -I {} mv {} $catImages/train/
    python imageCleaningAndGoogleSearching/clean.py $catImages/train
    numImages=$(ls -1 $catImages/all/ | wc -l)
    valImages=$(expr $numImages - $trainImages)
    gshuf -n $valImages -e $catImages/train/* | xargs -I {} mv {} $catImages/val/
    # split the datasets into 25 image groups
    for i in $(seq $numIncrements)
    do
        subsetCatImages=$catImages/$i_25
        rm -rf $subsetCatImages
        mkdir -p $subsetCatImages
        mkdir -p $subsetCatImages/noGoogleTrain
        mkdir -p $subsetCatImages/train
        mkdir -p $subsetCatImages/val

        gshuf -n $trainIncrements -e $catImages/train/* | xargs -I {} mv {} $subsetCatImages/noGoogleTrain/
        gshuf -n $valIncrements -e $catImages/val/* | xargs -I {} mv {} $subsetCatImages/val/
        python imageCleaningAndGoogleSearching/search.py $subsetCatImages/noGoogleTrain $subsetCatImages/train
    done
done





