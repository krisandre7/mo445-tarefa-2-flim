FLAGS= -O3 -Wall -Wno-unused-result -fPIC -fopenmp -pthread -std=gnu11 -pedantic -lsvm
CPP_FLAGS =  -Wall -Wno-unused-result  -Wno-long-long -fPIC -fopenmp -pthread -pedantic
LIBSVM = $(NEWIFT_DIR)/libsvm -lsvm -lstdc++
PYTHON_INCLUDE=/usr/include/python2.7
PYTHON_LIB=/usr/lib/python2.7
GC_DIR=./gc

all: graph_cut
	cd $(NEWIFT_DIR); make
	swig -python pyift.i
	gcc $(FLAGS) -Wno-unused-but-set-variable  -c pyift_wrap.c -I$(NEWIFT_DIR)/include -I$(PYTHON_INCLUDE)
	gcc $(FLAGS) -shared pyift_wrap.o -o _pyift.so -L$(NEWIFT_DIR)/lib -L$(LIBSVM) -L$(PYTHON_LIB) -lift -lm -llapack -lblas -lcblas
	swig -c++ -python pygc.i
	g++ $(CPP_FLAGS) -Wno-unused-but-set-variable -c pygc_wrap.cxx -I$(NEWIFT_DIR)/include -I$(PYTHON_INCLUDE) -I$(GC_DIR) 
	g++ $(CPP_FLAGS) -shared pygc_wrap.o -o _pygc.so -L$(NEWIFT_DIR)/lib -L$(LIBSVM) -L$(PYTHON_LIB) -L $(GC_DIR) -lift -lm -lgc -llapack -lblas -lcblas

graph_cut: 
	g++ -w $(CPP_FLAGS) -c $(GC_DIR)/graph.cpp -o $(GC_DIR)/obj/graph.o -I $(GC_DIR)
	g++ -w $(CPP_FLAGS) -c $(GC_DIR)/maxflow.cpp -o $(GC_DIR)/obj/maxflow.o -I $(GC_DIR)
	ar rvs $(GC_DIR)/libgc.a $(GC_DIR)/obj/*.o

clean:
	cd $(NEWIFT_DIR); make clean
