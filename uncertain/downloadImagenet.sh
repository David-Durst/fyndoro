#!/usr/bin/env bash
# first input is the wnid to download. Second input is the location to download it to
scriptDir=$(dirname "$(readlink -f "$0")")
wnid=$1
output_dir=$(readlink -f "$2")
name=$3
api_key=$(cat $scriptDir/../IMAGENET_API_KEY)

wget --directory-prefix $output_dir -O ${name}.tar "http://www.image-net.org/download/synset?wnid=${wnid}&username=durst&accesskey=${api_key}&release=latest&src=stanford"
mkdir -p $output_dir/$name
mv $output_dir/${name}.tar $output_dir/$name/
cd $output_dir/$name/
tar -xf ${name}.tar
rm ${name}.tar
cd -