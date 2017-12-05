#!/usr/bin/env bash
inputLocation1=$(dirname "$(readlink -f "$1")")
inputLocation2=$(dirname "$(readlink -f "$2")")
outputLocation1=$(dirname "$(readlink -f "$3")")
num=$5
IFS='
' # make it so that only the
cd $inputLocation1
filesToMove=$(shuf -n $num -e *)
for f in $filesToMove
do
    mv $f $outputLocation1
done

cd $inputLocation2
for f in $filesToMove
do
    rm $f
done