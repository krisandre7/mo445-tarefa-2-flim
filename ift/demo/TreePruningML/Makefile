BIN   =../../bin

#FLAGS= -ggdb -O0 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
FLAGS    = -O3 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic -g
IFTFLAGS = -I$(NEWIFT_DIR)/include

UNAME_S=$(shell uname -s)

DEBUG=0
ifeq ($(DEBUG), 1)
	FLAGS += -g
endif

ifeq ($(UNAME_S), Linux)
INCLUDES=-I$(NEWIFT_DIR)/include
IFTLIB=-L$(NEWIFT_DIR)/lib -lift -L/usr/lib/atlas-base -llapack -lblas -lcblas  -lm -lz -lpng
LIBSVM=-L$(NEWIFT_DIR)/libsvm -lsvm -lstdc++
CC=gcc
else
INCLUDES=-I$(NEWIFT_DIR)/include -I$(NEWIFT_DIR)/windows/libcblas/include
# Considering that IFT will be compiled using 64-bit MinGW. Also, cblas and lapack are provided inside IFT's directory
export CC=x86_64-w64-mingw32-gcc
IFTLIB    = -L$(NEWIFT_DIR)/lib -L$(NEWIFT_DIR)/windows/liblapack/lib -L$(NEWIFT_DIR)/windows/cblas/lib -lift -llapack -lblas  -lm
endif

LIBPNGINCLUDES = -I$(NEWIFT_DIR)/externals/libpng/include
LIBJPEGINCLUDES = -I$(NEWIFT_DIR)/externals/libjpeg/include
#LIBTIFFINCLUDES = -I$(NEWIFT_DIR)/externals/libtiff/include

LIBPNG = -L$(NEWIFT_DIR)/externals/libpng/lib -lpng
LIBJPEG = -L$(NEWIFT_DIR)/externals/libjpeg/lib -ljpeg
LIBTIFF = -L$(NEWIFT_DIR)/externals/libtiff/lib -ltiff

INCLUDES += $(LIBPNGINCLUDES) $(LIBJPEGINCLUDES) #$(LIBTIFFINCLUDES)

IFTLIB += $(LIBPNG) $(LIBJPEG) $(LIBTIFF)

ifeq ($(IFT_GPU), 1)
	export FLAGS += -DIFT_GPU=1
	INCLUDES += -I$(CUDA_DIR1)/include
	INCLUDES += -I$(CUDA_DIR2)/include
endif

ifeq ($(IFT_GPU),1)
   IFTLIB += -L$(CUDA_DIR)/lib64 -lcublas -lcudart
   INCLUDES += -I$(CUDA_DIR)/include
   FLAGS += -DIFT_GPU=1
endif

libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\


$@.c: libift $@.c
	make
#	$(CC) $(FLAGS) $(INCLUDES) $@.c -o $(BIN)/$@ $(IFTLIB) $(LIBSVM)
	$(CC) $(FLAGS) $(INCLUDES) $@.c -o $(BIN)/$@ $(IFTLIB) $(LIBSVM); \

clean:
	rm -f *.o $(BIN)/*;










