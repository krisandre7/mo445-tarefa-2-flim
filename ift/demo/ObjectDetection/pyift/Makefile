FLAGS= -O3 -Wall -Wno-unused-result -fPIC -fopenmp -pthread -std=gnu11 -pedantic
PYTHON_INCLUDE=/usr/include/python2.7
PYTHON_LIB=/usr/lib/python2.7

all:
	cd $(NEWIFT_DIR); make
	swig -python pyift.i 
	gcc $(FLAGS) -c pyift_wrap.c -I$(NEWIFT_DIR)/include -I$(PYTHON_INCLUDE)
	gcc $(FLAGS) -shared pyift_wrap.o -o _pyift.so -L$(NEWIFT_DIR)/lib -L$(PYTHON_LIB) -lift -lm -llapack -lblas -lcblas

clean:
	cd $(NEWIFT_DIR); make clean
