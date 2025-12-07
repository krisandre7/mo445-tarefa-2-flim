FLAGS= -O3 -Wall -Wno-unused-result -fPIC -fopenmp -pthread -std=gnu11 -pedantic -lsvm
CPP_FLAGS =  -w -fPIC -fopenmp -pthread -lsvm -llapack -lcblas -lblas
BIN=$(NEWIFT_DIR)/bin
LIBSVM = $(NEWIFT_DIR)/libsvm 

all: libift graph_cut

libift:
	@echo Building libift
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\

graph_cut:
	@echo Building libgc
	g++ $(CPP_FLAGS) -c graph.cpp -o obj/graph.o
	g++ $(CPP_FLAGS) -c maxflow.cpp -o obj/maxflow.o
	ar rvs libgc.a obj/graph.o obj/maxflow.o
	
gc_super: libift graph_cut
	g++ $(CPP_FLAGS) -c gc_super.cpp -o obj/gc_super.o -I $(NEWIFT_DIR)/include -L $(NEWIFT_DIR)/lib -L$(LIBSVM) -lgc -lift -lsvm -lstdc++
	g++ $(CPP_FLAGS) -o $(BIN)/gc_super obj/gc_super.o -I $(NEWIFT_DIR)/include -L $(NEWIFT_DIR)/lib -L. -L$(LIBSVM) -lgc -lift -lsvm -lstdc++
