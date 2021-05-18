#!/bin/bash
# this is a shell script to convert all .ipynb files to .py files in specified directory

set -e
set -o noclobber

echo "Conversion starts:"

if [ $# -eq 0 ]
  then
    dir="."
else
    dir=$1
fi

echo "Processing directory: $dir"

for ipynb_file in $(ls $dir/*.ipynb)
do
    ipynb_file="${ipynb_file%.ipynb}"
    echo "Processing file: $ipynb_file.ipynb"
    ipynb-py-convert $ipynb_file.ipynb $ipynb_file.py
    rm -f $ipynb_file.ipynb
done

echo "Conversion ends."
