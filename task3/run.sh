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
        runFile "./bin/t3_GPU.pg" "t3_GPU"
        ;;
    -all)
        shift
        runFile "./bin/t3_GPU.pg" "t3_GPU"
        ;;
    *)
        break
        ;;
  esac
done
exit 0