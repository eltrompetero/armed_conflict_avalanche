# ====================================================================================== #
# Script for compiling standalone, self-contained library for use in Python module.
# Author : Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
#!/bin/bash
PYCONFIGDR=/home/eddie/anaconda3/envs/acled2
PYTHONDR=/home/eddie/anaconda3/envs/acled2/bin

g++ -I$PYCONFIGDR/include/python3.8 -I$PYCONFIGDR/include \
    utils_ext.cpp -fpic -c -Ofast -o utils_ext.o && \
g++ -I$PYCONFIGDR/include/python3.8 -I$PYCONFIGDR/include \
    -Ofast \
    -L$PYCONFIGDR/lib -o utils_ext.so -shared utils_ext.o \
    -lpython3.8 -lboost_python38 -lboost_numpy38
