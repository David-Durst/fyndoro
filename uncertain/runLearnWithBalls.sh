#!/usr/bin/env bash
# note: fyndoro conda env must be active for this to work
scriptDir=$(dirname "$(readlink -f "$0")")
echo $scriptDir
imagesParent=$scriptDir/imagenet_balls
numImages=(25 50 75 100 125 150 175 200)

#get in directory above script for running uncertain.learn correctly
origDir=$(pwd)
cd $scriptDir/..

#make the output file
output_file=$scriptDir/balls_output.csv
rm -f $output_file
touch $output_file
echo "data_dir,num_labeled,best_val_acc" > $output_file

model_output_folder=$scriptDir/ballModels
rm -f $model_output_folder
mkdir -p $model_output_folder

for n in "${numImages[@]}"
do
    python -m uncertain.learn $imagesParent/augmented_${n}/ $n $output_file $model_output_folder
    python -m uncertain.learn $imagesParent/not_augmented_${n}/ $n $output_file $model_output_folder
done

cd $origDir
