#!/usr/bin/env bash
inputLocation1=$(readlink -f "$1")
inputLocation2=$(readlink -f "$2")
outputLocation1=$(readlink -f "$3")
num=$4
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