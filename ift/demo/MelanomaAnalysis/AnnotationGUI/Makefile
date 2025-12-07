FLAGS= -O3 -Wall -Wno-unused-result -fPIC -fopenmp -pthread -std=gnu11 -pedantic -lsvm
CPP_FLAGS =  -Wall -Wno-unused-result  -Wno-long-long -fPIC -fopenmp -pthread -pedantic
LIBSVM = $(NEWIFT_DIR)/libsvm -lsvm -lstdc++
PYTHON_INCLUDE=/usr/include/python2.7
PYTHON_LIB=/usr/lib/python2.7
GC_DIR=./gc

all: 
	cd $(NEWIFT_DIR); make
	swig -python pyift.i
	gcc $(FLAGS) -Wno-unused-but-set-variable  -c pyift_wrap.c -I$(NEWIFT_DIR)/include -I$(PYTHON_INCLUDE)
	gcc $(FLAGS) -shared pyift_wrap.o -o _pyift.so -L$(NEWIFT_DIR)/lib -L$(LIBSVM) -L$(PYTHON_LIB) -lift -lm -llapack -lblas 

clean:
	rm pyift_wrap.c pyift_wrap.o _pyift.so pyift.py pyift.pyc 
