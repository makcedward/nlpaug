#!/bin/bash

build_name='.'
pkg_name='nlpaug'
py_vers=(3.5 3.6 3.7 3.8)
pkg_ver='1.1.0dev'
conda_dir="~/anaconda3/envs/nlpaug_master/conda-bld"

echo "Building conda package ..."
for i in "${py_vers[@]}"
do
	conda-build --python $i $build_name
done

echo "Converting package to other platforms"
platforms=(osx-64 linux-32 win-32 win-64)
find "$conda_dir"/linux-64/"$pkg_name"*"$pkg_ver"*.tar.bz2 | while read file
do
	for platform in "${platforms[@]}"
	do
		conda convert --platform $platform $file -o "$conda_dir"
	done
done

echo "Upload to Anaconda"
for platform in "${platforms[@]}"
do
	find "$conda_dir"/"$platform"/"$pkg_name"*"$pkg_ver"*.tar.bz2 | while read file
	do
		anaconda upload --force $file
	done
done
