#!/bin/bash

build_name='.'
pkg_name='nlpaug'
py_vers=(3.5 3.6 3.7 3.8)
pkg_ver='0.0.20'
conda_dir="~/anaconda3/conda-bld/"

echo "Building conda package ..."
for i in "${py_vers[@]}"
do
	conda-build --python $i $build_name
done

echo "Converting package to other platforms"
platforms=(osx-64 linux-32 win-32 win-64)
find ~/anaconda3/conda-bld/linux-64/ -name *.tar.bz2 | while read file
do
	for platform in "${platforms[@]}"
	do
		conda convert --platform $platform $file -o ~/anaconda3/conda-bld/
	done
done

echo "Upload to Anaconda"
for platform in "${platforms[@]}"
do
	find ~/anaconda3/conda-bld/$platform/ -name *.tar.bz2 | while read file
	do
		anaconda upload --force $file
	done
done
