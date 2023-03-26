#!/bin/bash

pgc++ t3.cpp -o bin/t3_GPU.pg -fast -acc=gpu -O2 -D OPENACC__ -Mcudalib=cublas