#!/usr/bin/env bash
# first input is the wnid to download. Second input is the location to download it to
scriptDir=$(dirname "$(readlink -f "$0")")
wnid=$1
output_dir=$(readlink -f "$2")
api_key=$(cat $scriptDir/../IMAGENET_API_KEY)

wget --directory-prefix $output_dir "http://www.image-net.org/download/synset?wnid=${wnid}&username=durst&accesskey=${api_key}release=latest&src=stanford"