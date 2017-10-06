#!/usr/bin/env bash
# note, assume already have images into two folders, 1.0,0.0 and 0.0,1.0 for the two classes
# the absolute location of the folder containing those directories is the input
scriptDir=$(dirname "$(readlink -f "$0")")
set -x
images=$1
createTrainVal=false
createDatasets=true
categoryGroups=("1.0,0.0" "0.0,1.0")
googleCat="1.0,0.0"
googleCatOutput="0.8,0.2"
trainImages=2
# note that train increments need to sum to trainImages
trainIncrements=(1 1)
cat1Images=$(ls -1 $images/${categoryGroups[0]} | wc -l)
cat2Images=$(ls -1 $images/${categoryGroups[1]} | wc -l)
export MOZ_HEADLESS=1
if [ $cat1Images -gt $cat2Images ]
then
    valImages=$(expr $cat2Images - $trainImages)
else
    valImages=$(expr $cat1Images - $trainImages)
fi
valImages=10
numIncrements=${#trainIncrements[@]}

# get all the training and validation data cleaned and split up
if [ $createTrainVal == true ] ; then
    for c in "${categoryGroups[@]}"
    do
        rm -rf $images/train/$c
        rm -rf $images/val/$c
        mkdir -p $images/train/$c
        mkdir -p $images/val/$c
        # put all images for use in train first, then mv a subset of them to val
        shuf -n $(expr $valImages + $trainImages) -e $images/$c/* | xargs -I {} cp {} $images/train/$c/
        python $scriptDir/imageCleaningAndGoogleSearching/clean.py $images/train/$c
        shuf -n $valImages -e $images/train/$c/* | xargs -I {} mv {} $images/val/$c/
    done
fi

if [ $createDatasets == false ] ; then
    rm -rf $images/trainbackup
    rm -rf $images/valbackup
    cp -r $images/train $images/trainbackup
    cp -r $images/val $images/valbackup
    exit 0
fi

if [ $createDatasets == true ] && [ $createTrainVal == false ] ; then
    rm -rf $images/train
    rm -rf $images/val
    cp -r $images/trainbackup $images/train
    cp -r $images/valbackup $images/val
fi

sumIncrementsSoFar=0
# split the datasets into 25 image groups
for i in $(seq $numIncrements)
do
    subsetImages=$images/subset_${i}_of_${trainIncrements[$i]}_1.0,0.0_training_images
    rm -rf $subsetImages
    mkdir -p $subsetImages
    mkdir -p $subsetImages/train
    mkdir -p $subsetImages/val
    for c in "${categoryGroups[@]}"
    do
        mkdir -p $subsetImages/train/$c
        mkdir -p $subsetImages/val/$c
        # keep same validation images for every run
        shuf -n ${trainIncrements[$i]} -e $images/train/$c/* | xargs -I {} mv {} $subsetImages/train/$c/
        cp $images/val/$c/* $subsetImages/val/$c/
        if [ $c == "1.0,0.0" ]
        then
            rm -rf $subsetImages/train/0.8,0.2/
            mkdir -p $subsetImages/train/0.8,0.2/
            python $scriptDir/imageCleaningAndGoogleSearching/scrape.py $subsetImages/train/$c/ $subsetImages/train/0.8,0.2/ $scriptDir/../API_KEY
            python $scriptDir/imageCleaningAndGoogleSearching/clean.py $subsetImages/train/0.8,0.2/
        fi
        if [ $c == "0.0,1.0" ]
        then
            rm -rf $subsetImages/train/0.2,0.8/
            mkdir -p $subsetImages/train/0.2,0.8/
            python $scriptDir/imageCleaningAndGoogleSearching/scrape.py $subsetImages/train/$c/ $subsetImages/train/0.2,0.8/ $scriptDir/../API_KEY
            python $scriptDir/imageCleaningAndGoogleSearching/clean.py $subsetImages/train/0.2,0.8/
        fi
    done

    # join this with previous subsets to create training and validation runs of increasing size
    # this means that subset 1 leads to a merged subset of 25, subset 2 joins with subset 1 to make a merged subset of 50
    sumIncrementsSoFar=$(expr $sumIncrementsSoFar + ${trainIncrements[$i]})
    mergedSubset=$images/augmented_${sumIncrementsSoFar}
    mergedSubsetNoProb=${mergedSubset}_noprob
    mergedSubsetNoAugmentation=$images/not_augmented_${sumIncrementsSoFar}
    rm -rf $mergedSubset
    mkdir -p $mergedSubset
    cp -r $subsetImages/* $mergedSubset
    rm -rf $subsetImages
    if [ $i -gt 1 ]
    then
        cp -r $previousMergedSubset/* $mergedSubset
    fi
    rm -rf $mergedSubsetNoAugmentation
    mkdir -p $mergedSubsetNoAugmentation
    mkdir -p $mergedSubsetNoAugmentation/train
    mkdir -p $mergedSubsetNoAugmentation/val
    cp -r $mergedSubset/train/0.0,1.0 $mergedSubsetNoAugmentation/train/
    cp -r $mergedSubset/train/1.0,0.0 $mergedSubsetNoAugmentation/train/
    cp -r $mergedSubset/val/0.0,1.0 $mergedSubsetNoAugmentation/val/
    cp -r $mergedSubset/val/1.0,0.0 $mergedSubsetNoAugmentation/val/
    rm -rf $mergedSubsetNoProb
    cp -r $mergedSubset $mergedSubsetNoProb
    mv $mergedSubsetNoProb/train/0.2,0.8/* $mergedSubsetNoProb/train/0.0,1.0
    mv $mergedSubsetNoProb/train/0.8,0.2/* $mergedSubsetNoProb/train/1.0,0.0
    rmdir $mergedSubsetNoProb/train/0.2,0.8
    rmdir $mergedSubsetNoProb/train/0.8,0.2
    echo "Number of images in train 1.0,0.0"
    ls -1 $mergedSubset/train/1.0,0.0 | wc -l
    echo "Number of images in train 0.8,0.2"
    ls -1 $mergedSubset/train/0.8,0.2 | wc -l
    echo "Number of images in train 0.2,0.8"
    ls -1 $mergedSubset/train/0.2,0.8 | wc -l
    echo "Number of images in train 0.0,1.0"
    ls -1 $mergedSubset/train/0.0,1.0 | wc -l
    echo "Number of images in val 1.0,0.0"
    ls -1 $mergedSubset/val/1.0,0.0 | wc -l
    echo "Number of images in val 0.0,1.0"
    ls -1 $mergedSubset/val/0.0,1.0 | wc -l
    echo "Number of images in noprob train 1.0,0.0"
    ls -1 $mergedSubsetNoProb/train/1.0,0.0 | wc -l
    echo "Number of images in noprob train 0.0,1.0"
    ls -1 $mergedSubsetNoProb/train/0.0,1.0 | wc -l
    previousMergedSubset=$mergedSubset
done
