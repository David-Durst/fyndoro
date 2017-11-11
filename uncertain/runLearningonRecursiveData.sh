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
for i in $(seq $numIterations)
do
    outputFiles[$i]=$scriptDir/output/${outputName}_iter${i}.csv
    rm -f ${outputFiles[$i]}
    touch ${outputFiles[$i]}
    echo "data_dir,num_labeled,${i}_best_val_acc" > ${outputFiles[$i]}
done

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
        num_t0=$(ls -1 $uptoiDir/train/${categoryGroups[0]}/ | wc -l)
        num_t1=$(ls -1 $uptoiDir/train/${categoryGroups[1]}/ | wc -l)
        num_total=${num_t0}_${num_t1}
        num_training=$(expr 2 \* $n)
        num_str=${num_training}_${num_total}
        python -m uncertain.learn $uptoiDir $num_str ${outputFiles[$i]} $model_output_folder
     done
done

cd $origDir
