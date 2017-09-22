#!/usr/bin/env bash
# note, assume already have images into two folders, 1.0,0.0 and 0.0,1.0 for the two classes
# the location of the folder containing those directories is the input
scriptDir=$(dirname "$(readlink -f "$0")")
echo $scriptDir
images=$scriptDir/$1
categoryGroups=("1.0,0.0" "0.0,1.0")
googleCat="1.0,0.0"
googleCatOutput="0.8,0.2"
trainImages=200
trainIncrements=25
cat1Images=$(ls -1 $images/${categoryGroups[0]} | wc -l)
cat2Images=$(ls -1 $images/${categoryGroups[1]} | wc -l)
if [ $cat1Images > $cat2Images ]
then
    valImages=$(expr $cat2Images - $trainImages)
else
    valImages=$(expr $cat2Images - $trainImages)
fi
numIncrements=$(expr $trainImages / $trainIncrements)

# get all the training and validation data cleaned and split up
for c in "${categoryGroups[@]}"
do
    rm -rf $images/train/$c
    rm -rf $images/val/$c
    mkdir -p $images/train/$c
    mkdir -p $images/val/$c
    # put all images for use in train first, then mv a subset of them to val
    shuf -n $(expr $valImages + $trainImages) -e $images/$c/* | xargs -I {} cp {} $images/train/$c/
    python imageCleaningAndGoogleSearching/clean.py $images/train/$c
    shuf -n $valImages -e $images/train/$c/* | xargs -I {} mv {} $images/val/$c/
done

# split the datasets into 25 image groups
for i in $(seq $numIncrements)
do
    subsetImages=$images/subset_${i}_of_${trainIncrements}_1.0,0.0_training_images
    rm -rf $subsetImages
    mkdir -p $subsetImages
    mkdir -p $subsetImages/train
    mkdir -p $subsetImages/val
    for c in "${categoryGroups[@]}"
    do
        mkdir -p $subsetImages/train/$c
        mkdir -p $subsetImages/val/$c
        # keep same validation images for every run
        shuf -n $trainIncrements -e $images/train/$c/* | xargs -I {} mv {} $subsetImages/train/$c/
        cp $images/val/$c/* $subsetImages/val/$c/
        if [ $c == "1.0,0.0" ]
        then
            rm -rf $subsetImages/train/0.8,0.2/
            mkdir -p $subsetImages/train/0.8,0.2/
            python $scriptDir/imageCleaningAndGoogleSearching/search.py $subsetImages/train/$c/ $subsetImages/train/0.8,0.2/
            python $scriptDir/imageCleaningAndGoogleSearching/clean.py $subsetImages/train/0.8,0.2/
        fi
        if [ $c == "0.0,1.0" ]
        then
            rm -rf $subsetImages/train/0.2,0.8/
            mkdir -p $subsetImages/train/0.2,0.8/
            python $scriptDir/imageCleaningAndGoogleSearching/search.py $subsetImages/train/$c/ $subsetImages/train/0.2,0.8/
            python $scriptDir/imageCleaningAndGoogleSearching/clean.py $subsetImages/train/0.2,0.8/
        fi
    done

    # join this with previous subsets to create training and validation runs of increasing size
    # this means that subset 1 leads to a merged subset of 25, subset 2 joins with subset 1 to make a merged subset of 50
    mergedSubset=$images/augmented_$(expr $trainIncrements "*" $i)
    mergedSubsetNoAugmentation=$images/not_augmented_$(expr $trainIncrements "*" $i)
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
    previousMergedSubset=$mergedSubset
done
