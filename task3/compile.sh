#!/bin/bash

pgc++ t3.cpp -o t3_GPU.pg -fast -acc=gpu -O2 -D OPENACC__ -D NVPROF_ -Mcudalib=cublas