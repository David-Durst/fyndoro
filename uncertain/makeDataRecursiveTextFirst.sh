#!/usr/bin/env bash
set -x
# note, assume already have images in two folders with names from categoryGroups variable for the two classes
# the absolute location of the folder containing those directories is the input
scriptDir=$(dirname "$(readlink -f "$0")")
images=$1
createTrainVal=true
createDatasets=true
categoryGroups=("scarletTanager" "summerTanager")
declare -A searchwords=( [${categoryGroups[0]}]="scarlet tanager" [${categoryGroups[1]}]="summer tanager")
declare -A keywordFilters=( [${categoryGroups[0]}]="scarlet" [${categoryGroups[1]}]="summer")
# wrongwords are words that indicate an image is bad
declare -A wrongwordFilters=( [${categoryGroups[0]}]="summer" [${categoryGroups[1]}]="scarlet")
numIterations=3
# setting train images to 0 here as downloading all train images by label
trainImages=0
# note that train increments need to sum to trainImages
cat1Images=$(ls -1 $images/${categoryGroups[0]} | wc -l)
cat2Images=$(ls -1 $images/${categoryGroups[1]} | wc -l)
export MOZ_HEADLESS=1
if [ $cat1Images -gt $cat2Images ]
then
    valImages=$(expr $cat2Images - $trainImages)
else
    valImages=$(expr $cat1Images - $trainImages)
fi

# get all the training and validation data cleaned and split up
if [ $createTrainVal == true ] ; then
    for c in "${categoryGroups[@]}"
    do
        rm -rf $images/train/$c
        rm -rf $images/val/$c
        mkdir -p $images/train/$c
        mkdir -p $images/val/$c
        # put all images for use in train first, then mv a subset of them to val
        set +x
        shuf -n $(expr $valImages + $trainImages) -e $images/$c/* | xargs -I {} cp {} $images/train/$c/
        set -x
        python $scriptDir/imageCleaningAndGoogleSearching/clean.py $images/train/$c
        set +x
        shuf -n $valImages -e $images/train/$c/* | xargs -I {} mv {} $images/val/$c/
        set -x
    done
fi

if [ $createDatasets == false ] || [ ! -d $images/trainbackup ] ; then
    rm -rf $images/trainbackup
    rm -rf $images/valbackup
    cp -r $images/train $images/trainbackup
    cp -r $images/val $images/valbackup
    if [ $createDatasets == false ] ; then
        exit 0
    fi
fi

if [ $createDatasets == true ] && [ $createTrainVal == false ] ; then
    rm -rf $images/train
    rm -rf $images/val
    cp -r $images/trainbackup $images/train
    cp -r $images/valbackup $images/val
fi

downloadedImages=$images/downloadedImages
# remove all old training and validation copies of data
rm -rf $downloadedImages
mkdir -p $downloadedImages
for c in "${categoryGroups[@]}"
do
    for j in $(seq $numIterations)
    do
        curIterDir=$downloadedImages/$j/train/$c
        nextIterDir=$downloadedImages/$(expr $j + 1)/train/$c
        uptoCurDir=$downloadedImages/upto_$j/train/$c
        uptoNextIterDir=$downloadedImages/upto_$(expr $j + 1)/train/$c
        mkdir -p $curIterDir
        # make the next one as well, as that is where the script puts scrape results in
        mkdir -p $nextIterDir
        # keep same validation images for every run
        # only need to make val for cur dir if this is first iteration, other curs will be made by prev iteration
        curIterValDir=$downloadedImages/$j/val
        nextIterValDir=$downloadedImages/$(expr $j + 1)/val
        uptoCurValDir=$downloadedImages/upto_$j/val
        uptoNextIterValDir=$downloadedImages/upto_$(expr $j + 1)/val
        if [ $j == 1 ]
        then
            mkdir -p $curIterValDir
            ln -sr $images/val/$c $curIterValDir
            mkdir -p $uptoCurValDir
            ln -sr $images/val/$c $uptoCurValDir
        fi
        mkdir -p $nextIterValDir
        ln -sr $images/val/$c $nextIterValDir
        mkdir -p $uptoNextIterValDir
        ln -sr $images/val/$c $uptoNextIterValDir
        # if first time scraping for this category group, create
        # first set of images by scraping by text
        if [ $j == 1 ]
        then
            set +x
            python $scriptDir/imageCleaningAndGoogleSearching/textScrape.py "${searchwords[$c]}" $curIterDir/
            python $scriptDir/imageCleaningAndGoogleSearching/clean.py $curIterDir/
            set -x
        fi
        python $scriptDir/imageCleaningAndGoogleSearching/scrape.py $curIterDir/ $nextIterDir/ "${keywordFilters[$c]}" "${wrongwordFilters[$c]}"
        python $scriptDir/imageCleaningAndGoogleSearching/clean.py $nextIterDir/
        # join next iter images with all previous ones
        mkdir -p $uptoNextIterDir
        # if first iter, make upToCurDir as no prev iteration to make it
        if [ $j == 1 ]
        then
            mkdir -p $uptoCurDir
            rsync --ignore-existing -a $curIterDir/ $uptoCurDir/
        fi
        rsync --ignore-existing -a $nextIterDir/ $uptoNextIterDir/
        rsync --ignore-existing -a $uptoCurDir/ $uptoNextIterDir/
    done
done


echo "Number of images in " $downloadedImages
# https://unix.stackexchange.com/questions/4105/how-do-i-count-all-the-files-recursively-through-directories
set +x
find $downloadedImages -type d -links 2 | while read -r dir; do printf "%s:\t" "$dir"; find "$dir" -type f | wc -l; done
set -x
