#!/usr/bin/env bash
# note: fyndoro conda env must be active for this to work
# argument 1 is the location of the images. argument 2 is the name to use for output files and folders
imagesParent=$(readlink -f "$1")
outputName=$2
scriptDir=$(dirname "$(readlink -f "$0")")
numImages=(1 2 5)
numIterations=4
#$scriptDir/makeDataForExperiments.sh $imagesParent

#get in directory above script for running uncertain.learn correctly
origDir=$(pwd)
cd $scriptDir/..

#make the output files
outputFiles=()
for i in $(seq $numIterations)
do
    outputFiles[$i]=$scriptDir/output/${outputName}_iter${i}.csv
    rm -f ${outputFiles[$i]}
    touch ${outputFiles[$i]}
    echo "data_dir,num_labeled,${i}_best_val_acc" > ${outputFiles[$i]}
done

model_output_folder=$scriptDir/${outputName}Models
rm -rf $model_output_folder
mkdir -p $model_output_folder

for n in "${numImages[@]}"
do
    echo "Running experiments for merged_$n"
    for i in $(seq $numIterations)
    do
        num_t1=$(ls -1 $imagesParent/merged_${n}/upto_${i}/train/1.0,0.0/ | wc -l)
        num_t0=$(ls -1 $imagesParent/merged_${n}/upto_${i}/train/0.0,1.0/ | wc -l)
        num_total=${num_t1}_${num_t0}
        num_training=$(expr 2 \* $n)
        num_str=${num_training}_${num_total}
        python -m uncertain.learn $imagesParent/merged_${n}/upto_${i} $num_str ${outputFiles[$i]}
     done
done

cd $origDir
