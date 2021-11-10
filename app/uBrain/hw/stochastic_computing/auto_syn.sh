#!/bin/bash

set -e
set -o noclobber

alias cp="cp -i"
unalias cp

technode="32nm_rvt"
# technode="32nm_hvt"

echo "Process technode: $technode"

if [ ! -d "./$technode" ] 
then
    mkdir $technode
    echo "mkdir $technode"
fi

cp -R rtl/*.sv ./$technode

cd $technode
echo "Process dir: $technode"
rm -rf syn* .syn*
cp ../../../syn_* .
cp ../../../synopsys_dc.setup_$technode .synopsys_dc.setup
source syn_run.sh
cd ..
