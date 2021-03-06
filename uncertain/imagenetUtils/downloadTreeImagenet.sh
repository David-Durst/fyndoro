#!/usr/bin/env bash
scriptDir=$(dirname "$(readlink -f "$0")")
wnid=$1
output_dir=$(readlink -f "$2")
name=$3

treeFile=${name}tree.txt
wget -O $treeFile "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=${wnid}&full=1"
#get rid of starting - and any trailing spaces in each line
sed -i 's/-//g' $treeFile
sed -i 's/\r//g' $treeFile
sed -i 's/[[:blank:]]*$//g' $treeFile
while read wnid; do
  ${scriptDir}/downloadImagenet.sh $wnid $2/$name $wnid
done < $treeFile