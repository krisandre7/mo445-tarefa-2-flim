BIN=../../bin
NEWIFT_DIR=../../

#FOR GPU support uncomment the next line (in all the Makefiles that will use GPU)
IFT_GPU=1

CC = gcc
CXX = g++
IDIR := include
SDIR := src
ODIR := obj

#TensorFlow C API flags
TFLIB = -ltensorflow

#cppflow 
CPPFLOW_SDIR := cppflow/src
CPPFLOW_IDIR := cppflow/include

FLAGS= -O3 -DNDEBUG -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic -I$(IDIR)
#FLAGS= -pg -g -ggdb -Wall -DNDEBUG -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic -I$(IDIR)
CPP_FLAGS= -O3 -DNDEBUG -Wall -Wno-unused-result -fopenmp -pthread -std=c++17 -pedantic -lstdc++ -I$(IDIR) -I$(CPPFLOW_IDIR)
#CPP_FLAGS= -pg -g -ggdb -Wall -Wno-unused-result -fopenmp -pthread -std=c++17 -pedantic -lstdc++ -I$(IDIR) -I$(CPPFLOW_IDIR)
IFTLIB    = -L$(NEWIFT_DIR)/lib -L$(HOME)/atlas/lib -lift -llapack -lblas -latlas -lm
LIBSVM    = -L$(NEWIFT_DIR)/externals/libsvm/lib -lsvm -lstdc++ 
IFTFLAGS  = -I$(NEWIFT_DIR)/include -I$(HOME)/atlas/include -I$(NEWIFT_DIR)/externals/libsvm/include
libift = $(NEWIFT_DIR)/lib/libift.a

#CUDA path in case IFT_GPU is enabled
export CUDA_DIR1=/usr/local/cuda
export CUDA_DIR2=/opt/cuda


ifeq ($(IFT_GPU), 1)
	export FLAGS += -DIFT_GPU=1
	IFTFLAGS += -I $(CUDA_DIR1)/include
	IFTFLAGS += -I $(CUDA_DIR2)/include
	IFTLIB += -L $(CUDA_DIR1)/lib64 -L $(CUDA_DIR2)/lib64 -lcublas -lcudart
    MAKE_CMD=$(MAKE) IFT_GPU=1
else
    MAKE_CMD=$(MAKE)
endif

#shared includes
_DEPS = iftHybridNetwork.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

#shared objects
_OBJ = iftHybridNetwork.o \
       iftHybridNetworkTF.o \
       Tensor.o \
       Model.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: $(libift)
	$(MAKE) iftGetSVMErrorMatrix
	$(MAKE) iftGetSVMOptimalParameters
	$(MAKE) iftHybridExperiment
	$(MAKE) iftHybridExperimentTF
	$(MAKE) iftNetworkBatchExperiment
	$(MAKE) iftNetworkPreProcImgs_iftStd
	$(MAKE) iftNetworkPreProcImgs_celsoStd
	$(MAKE) iftPartitionSVMDataset
	$(MAKE) iftSVMExperiment
	$(MAKE) iftUpdateDataSetFileSetDir
	$(MAKE) iftUpdateSampleCsvDir
	$(MAKE) iftCheckTFNetwork
	$(MAKE) iftNetworkBatchExperimentTF

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS) $(libift)
	$(CC) $(FLAGS) -c -o $@ $< $(IFTFLAGS) $(IFTLIB) $(LIBSVM) 

$(ODIR)/Model.o: 
	$(CXX) $(CPP_FLAGS) -c -o $@ $(CPPFLOW_SDIR)/Model.cpp $(TFLIB)

$(ODIR)/Tensor.o: 
	$(CXX) $(CPP_FLAGS) -c -o $@ $(CPPFLOW_SDIR)/Tensor.cpp $(TFLIB)

$(ODIR)/iftHybridNetworkTF.o: $(SDIR)/iftHybridNetworkTF.cpp
	$(CXX) $(CPP_FLAGS) -c -o $@ $< $(IFTFLAGS) $(IFTLIB) $(LIBSVM) $(TFLIB)

$(libift):
	$(MAKE) -C $(NEWIFT_DIR); cd -\

$@.c: $(libift) $@.c $(OBJ)
	$(MAKE) $(OBJ)
	$(CC) $(FLAGS) $@.c -o $(BIN)/$@ $(OBJ) $(IFTFLAGS) $(IFTLIB) $(LIBSVM) $(TFLIB); \

$@.cpp: $(libift) $@.cpp $(OBJ)
	$(MAKE) $(OBJ)
	$(CXX) $(CPP_FLAGS) $@.cpp -o $(BIN)/$@ $(OBJ) $(IFTFLAGS) $(IFTLIB) $(LIBSVM) $(TFLIB); \



.PHONY: clean
clean:
	rm -f $(ODIR)/*.o
	rm -df $(ODIR)
	rm -f $(BIN)/*;

$(OBJ): | $(ODIR)

$(ODIR):
	mkdir $(ODIR)
