#!/bin/bash

if [ -d "build" ]; then
    if [ -z "$1" ]; then
        rm -rf build
        mkdir build
    fi
else
    mkdir build
fi

cd build

nvcc -c ../MST/*.cu -lcublas

nvcc -c ../explorer.cu -lcublas

nvcc -lcublas *.o -o t6