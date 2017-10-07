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
output_file_augmented=$scriptDir/output/${outputName}_augmented.csv
output_file_noprob=$scriptDir/output/${outputName}_noprob.csv
output_file_notaugmented=$scriptDir/output/${outputName}_notaugmented.csv
rm -f $output_file_augmented
rm -f $output_file_noprob
rm -f $output_file_notaugmented
touch $output_file_augmented
touch $output_file_noprob
touch $output_file_notaugmented
echo "data_dir,num_labeled,best_val_acc" > $output_file_augmented
echo "data_dir,num_labeled,best_val_acc" > $output_file_noprob
echo "data_dir,num_labeled,best_val_acc" > $output_file_notaugmented

model_output_folder=$scriptDir/${outputName}Models
rm -rf $model_output_folder
mkdir -p $model_output_folder

for n in "${numImages[@]}"
do
    num_t1=$(ls -1 $imagesParent/augmented_${n}_noprob/train/1.0,0.0/)
    num_t0=$(ls -1 $imagesParent/augmented_${n}_noprob/train/0.0,1.0/)
    num_total=$(expr $num_t0 + $num_t1)
    num_training=$(expr 2 \* $n)
    num_str=$num_total-$num_training
    python -m uncertain.learn $imagesParent/augmented_${n}/ $num_str $output_file_augmented $model_output_folder
    python -m uncertain.learn $imagesParent/augmented_${n}_noprob/ $num_str $output_file_noprob $model_output_folder
    python -m uncertain.learn $imagesParent/not_augmented_${n}/ $num_str $output_file_notaugmented $model_output_folder
done

cd $origDir
