#!/usr/bin/env bash
# note: fyndoro conda env must be active for this to work
# argument 1 is the location of the images. argument 2 is the name to use for output files and folders
imagesParent=$(readlink -f "$1")
outputName=$2
scriptDir=$(dirname "$(readlink -f "$0")")
numImages=(25 50 75 100 125 150 175 200)
$scriptDir/makeDataForExperiments.sh $imagesParent

#get in directory above script for running uncertain.learn correctly
origDir=$(pwd)
cd $scriptDir/..

#make the output file
output_file=$scriptDir/output/${outputName}.csv
rm -f $output_file
touch $output_file
echo "data_dir,num_labeled,best_val_acc" > $output_file

model_output_folder=$scriptDir/${outputName}Models
rm -rf $model_output_folder
mkdir -p $model_output_folder

for n in "${numImages[@]}"
do
    python -m uncertain.learn $imagesParent/augmented_${n}/ $n $output_file $model_output_folder
    python -m uncertain.learn $imagesParent/not_augmented_${n}/ $n $output_file $model_output_folder
done

cd $origDir
