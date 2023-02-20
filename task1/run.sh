#!/bin/bash

function runFile {
    echo -n "   Run "
    echo $2
    echo -n ""
    $1
    echo -e "----"
    echo -e ""
}

while test $# -gt 0; do
    case "$1" in 
    -gen)
        shift
        ./compile.sh
        ;;
    -gpu)
        shift
        runFile "nvprof ./t1_Double_GPU.pg" "t1_Double_GPU"
        runFile "nvprof ./t1_Float_GPU.pg" "t1_Float_GPU"
        ;;
    -mp)
        shift
        re='^[0-9]+$'
        if [[ $1 =~ $re ]];
        then
            echo -n "Recompile with "
            echo -n $1
            echo " threads"
            g++ task1_openMP.cpp -o t1_Double_MultiCore.mp -fopenmp -lpthread -D CPU -D Double -D nThreads=$1
            g++ task1_openMP.cpp -o t1_Float_MultiCore.mp -fopenmp -lpthread -D CPU -D nThreads=$1
        fi
        runFile "./t1_Double_MultiCore.mp" "t1_Double_MultiCore MP"
        runFile "./t1_Float_MultiCore.mp" "t1_Float_MultiCore MP"
        ;;
    -pgMult)
        shift
        runFile "nvprof ./t1_Double_MultiCore.pg" "t1_Double_MultiCore PGC"
        runFile "nvprof ./t1_Float_MultiCore.pg" "t1_Float_MultiCore PGC"
        ;;
    -pgOne)
        shift
        runFile "./t1_Double_OneCore.pg" "t1_Double_OneCore PGC"
        runFile "./t1_Float_OneCore.pg" "t1_Float_OneCore PGC"
        ;;
    -all)
        runFile "nvprof ./t1_Double_GPU.pg" "t1_Double_GPU"
        runFile "nvprof ./t1_Float_GPU.pg" "t1_Float_GPU"
        runFile "nvprof ./t1_Double_MultiCore.pg" "t1_Double_MultiCore PGC"
        runFile "nvprof ./t1_Float_MultiCore.pg" "t1_Float_MultiCore PGC"
        runFile "./t1_Double_MultiCore.mp" "t1_Double_MultiCore MP"
        runFile "./t1_Float_MultiCore.mp" "t1_Float_MultiCore MP"
        runFile "./t1_Double_OneCore.pg" "t1_Double_OneCore PGC"
        runFile "./t1_Float_OneCore.pg" "t1_Float_OneCore PGC"
        ;;
    *)
        break
        ;;
  esac
done
exit 0