#!/bin/bash

pgc++ t4.cu -o bin/t4_GPU -O2 -Mcuda -fast -ta=tesla:cc70