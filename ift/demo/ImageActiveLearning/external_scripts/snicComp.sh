#!/bin/sh
g++ snic.cpp -lm -lift -L../../lib -I../../include -I../../externals/libsvm/include -o ../../bin/snic -std=c++11 -lpthread -fopenmp -lblas -llapack
