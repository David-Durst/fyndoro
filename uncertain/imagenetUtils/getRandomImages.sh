#!/usr/bin/env bash
input_dir=$(readlink -f "$1")
output_dir=$(readlink -f "$2")
num_to_copy=$3

for d in ${input_dir}/*/; do
    shuf -n $num_to_copy -e $d/* | xargs -I {} cp {} $output_dir
done
