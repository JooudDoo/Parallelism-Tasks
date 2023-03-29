#!/bin/bash

timed=0

function runFile {
    echo -n "   Run "
    echo $2
    echo -n "";
    if (($timed != 0)); then
        echo -n "Executing (Timed): "
        echo ${prof}${1}${flags}
        time (${prof}${1}${flags})
    else
        echo -n "Executing: "
        echo ${prof}${1}${flags}
        ${prof}${1}${flags}
    fi
    echo -e "----"
    echo -e ""
}

while test $# -gt 0; do
    case "$1" in 
    -prof)
        shift
        prof="$1 "
        shift
        ;;
    -fl)
        shift
        flags=" $1"
        shift
        ;;
    -timed)
        shift
        timed=1
        ;;
    -gen)
        shift
        ./compile.sh
        ;;
    -gpu)
        shift
        runFile "./bin/t2_GPU.pg" "t2_GPU"
        ;;
    -pgMult)
        shift
        runFile "./bin/t2_MultiCore.pg" "t2_MultiCore PGC"
        ;;
    -pgOne)
        shift
        runFile "./bin/t2_OneCore.pg" "t2_OneCore PGC"
        ;;
    -all)
        shift
        runFile "./bin/t2_GPU.pg" "t2_GPU"
        runFile "./bin/t2_MultiCore.pg" "t2_MultiCore PGC"
        runFile "./bin/t2_OneCore.pg" "t2_OneCore PGC"
        ;;
    *)
        break
        ;;
  esac
done
exit 0