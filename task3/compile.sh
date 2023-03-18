#!/bin/bash

pgc++ t3.cpp -o t3_MultiCore.pg -fast -acc=multicore -O2 -Mconcur=allcores -Mcudalib=cublas

pgc++ t3.cpp -o t3_OneCore.pg -fast -O2 -Mcudalib=cublas

pgc++ t3.cpp -o t3_GPU.pg -fast -acc=gpu -O2 -D OPENACC__ -D NVPROF_ -Mcudalib=cublas