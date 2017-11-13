#!/usr/bin/env bash
# note: fyndoro conda env must be active for this to work
# argument 1 is the location of the images. argument 2 is the name to use for output files and folders
imagesParent=$(readlink -f "$1")
outputName=$2
scriptDir=$(dirname "$(readlink -f "$0")")
categoryGroups=("scarletTanager" "summerTanager")
numImages=(1 2 5)
numIterations=4
#$scriptDir/makeDataForExperiments.sh $imagesParent

#get in directory above script for running uncertain.learn correctly
origDir=$(pwd)
cd $scriptDir/..

#make the output files
outputFiles=()
# the accuracy of each iteration
for i in $(seq $numIterations)
do
    outputFiles[$i]=$scriptDir/output/${outputName}_iter${i}.csv
    rm -f ${outputFiles[$i]}
    touch ${outputFiles[$i]}
    times_augmented=$(expr $i - 1)
    echo "data_dir,number of training images from imagenet,$times_augmented times augmented" > ${outputFiles[$i]}
done
# the number of images in each iteration
numImagesFile=$scriptDir/output/${outputName}_numimages.csv
num_images_header="data_dir,number of times augmented,ImageNet images"
for categoryGroup in "${categoryGroups[@]}"
do
    num_images_header=$num_images_header,${categoryGroup}" total images"
done
echo $num_images_header > $numImagesFile

for i in $(seq $numIterations)
do
    model_output_folder=$scriptDir/${outputName}Models_${i}
    rm -rf $model_output_folder
    mkdir -p $model_output_folder
done

for n in "${numImages[@]}"
do
    echo "Running experiments for merged_$n"
    for i in $(seq $numIterations)
    do
        uptoiDir=$imagesParent/merged_${n}/upto_${i}
        python uncertain/getEmbeddingsForData.py $uptoiDir/ "" $uptoiDir/embeddings
        python uncertain/imageCleaningAndGoogleSearching/filterBasedOnEmbeddingSimilarity.py $uptoiDir/embeddings
        rm $uptoiDir/embeddings
        num_imagenet=$(expr 2 \* $n)
        python -m uncertain.learn $uptoiDir $num_imagenet ${outputFiles[$i]} $model_output_folder
        num_augmented_images=${uptoiDir},$(expr $i - 1),${num_imagenet}
        set -x
        for categoryGroup in "${categoryGroups[@]}"
        do
            num_in_category_group=$(ls -1 $uptoiDir/train/${categoryGroup}/ | wc -l)
            num_augmented_images=$num_augmented_images,$num_in_category_group
        done
        set +x
        echo $num_augmented_images >> $numImagesFile
     done
done

cd $origDir
