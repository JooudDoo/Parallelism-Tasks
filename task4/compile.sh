#!/bin/bash

# pgc++ t4.cu -o bin/t4_GPU -O2 -Mcuda -acc -ta=tesla:cc61 -fast
/usr/local/cuda/bin/nvcc -rdc=true -O2 t4.cu -o bin/t4_GPU