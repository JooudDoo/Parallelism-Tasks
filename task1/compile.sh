#!/bin/bash

pgc++ task1.cpp -o t1_Double_MultiCore.pg -fast -acc=multicore -D CPU -D Double
pgc++ task1.cpp -o t1_Float_MultiCore.pg -fast -acc=multicore -D CPU

g++ task1_openMP.cpp -o t1_Double_MultiCore.mp -fopenmp -lpthread -D CPU -D Double -D nThreads=20
g++ task1_openMP.cpp -o t1_Float_MultiCore.mp -fopenmp -lpthread -D CPU -D nThreads=20

pgc++ task1.cpp -o t1_Double_OneCore.pg -fast -D CPU -D Double
pgc++ task1.cpp -o t1_Float_OneCore.pg -fast -D CPU

pgc++ task1.cpp -o t1_Double_GPU.pg -fast -acc=gpu -D Double
pgc++ task1.cpp -o t1_Float_GPU.pg -fast -acc=gpu