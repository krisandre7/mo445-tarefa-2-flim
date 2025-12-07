BIN=../../bin
OBJ=obj

#FLAGS= -g -O0 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
IFTLIB= -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lm -L/lib64/atlas-sse3/ -L/usr/lib/atlas-base -lcblas -lpng -lz
LIBSVM= -L$(NEWIFT_DIR)/externals/libsvm/lib -lsvm -lstdc++
INCLUDES= -I$(NEWIFT_DIR)/include -I$(NEWIFT_DIR)/externals/libsvm/include

IFT_DEBUG=0
ifeq ($(IFT_DEBUG), 1)
	FLAGS += -pg -g -fsanitize=address -fsanitize=leak -DIFT_DEBUG=1
else
    FLAGS += -O3
endif

#CUDA path in case IFT_GPU is enabled
export CUDA_DIR1=/usr/local/cuda
export CUDA_DIR2=/opt/cuda

IFT_GPU=0
ifeq ($(IFT_GPU), 1)
	export FLAGS += -DIFT_GPU=1
	INCLUDES += -I$(CUDA_DIR1)/include
	INCLUDES += -I$(CUDA_DIR2)/include
	IFTLIB   += -L$(CUDA_DIR1)/lib64 -L$(CUDA_DIR2)/lib64 -lcublas -lcudart
endif


libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\


$@.c: libift $@.c
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(INCLUDES) $(IFTLIB) $(LIBSVM); \

clean:
	rm $(BIN)/*;










