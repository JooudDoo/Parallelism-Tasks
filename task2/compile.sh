#!/bin/bash

pgc++ t2.cpp -o t2_MultiCore.pg -fast -acc=multicore -O2 -Mconcur=allcores

pgc++ t2.cpp -o t2_OneCore.pg -fast -O2

pgc++ t2.cpp -o t2_GPU.pg -fast -acc=gpu -O2 -D OPENACC__ -D NVPROF_