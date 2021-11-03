#!/bin/bash

set -e
set -o noclobber

alias cp="cp -i"
unalias cp

technode="32nm_rvt"

echo "Process technode: $technode"

cp -R ./src_sv/* ./$technode

cd $technode

echo "Process dir: $technode"
rm -rf syn* .syn*
cp ../syn_* .
cp ../synopsys_dc.setup_$technode .synopsys_dc.setup
source syn_run.sh

cd ..
