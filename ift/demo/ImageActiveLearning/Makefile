BIN=../../bin
NEWIFT_DIR=../../

#FOR GPU support uncomment the next line (in all the Makefiles that will use GPU)
#IFT_GPU=1

CC = gcc
IDIR := include
SDIR := src
ODIR := obj

#FLAGS= -pg -g -ggdb -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic -I$(IDIR)
FLAGS= -O3 -DNDEBUG -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic -I$(IDIR)
#IFTLIB    = -L$(NEWIFT_DIR)/lib  -lift -llapack -lblas -lm -lpng -ljpeg -lz
#IFTLIB    = -L$(NEWIFT_DIR)/lib -L$(HOME)/atlas/lib -lift -llapack -lblas -latlas -lm -L$(NEWIFT_DIR)/externals/METIS/lib -lmetis
IFTLIB    = -L$(NEWIFT_DIR)/lib -L$(HOME)/atlas/lib -lift -llapack -lblas -latlas -lm
LIBSVM    = -L$(NEWIFT_DIR)/externals/libsvm/lib -lsvm -lstdc++ 
IFTFLAGS  = -I$(NEWIFT_DIR)/include -I$(HOME)/atlas/include -I$(NEWIFT_DIR)/externals/libsvm/include -I$(NEWIFT_DIR)/externals/METIS/include
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
_DEPS = iftBoVW.h \
	iftRandomSelector.h \
	iftUnlabeledSelector.h \
	iftActiveClassifier.h \
	iftListAppend.h \
	iftSuperpixelFeatures.h \
	iftSuperpixelFeatureList.h \
	iftExperimentUtility.h \
	iftMultiScaleGraph.h \
	iftHybridNetwork.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

#shared objects
_OBJ = iftBoVW.o \
       iftRandomSelector.o \
       iftUnlabeledSelector.o \
       iftActiveClassifier.o \
       iftListAppend.o \
       iftSuperpixelFeatures.o \
       iftSuperpixelFeatureList.o \
       iftExperimentUtility.o \
       iftMultiScaleGraph.o \
       iftHybridNetwork.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS) $(libift)
	$(CC) $(FLAGS) -c -o $@ $< $(IFTFLAGS) $(IFTLIB) $(LIBSVM) 

$(libift):
	$(MAKE) -C $(NEWIFT_DIR); cd -\

$@.c: $(libift) $@.c $(OBJ)
	$(MAKE) $(OBJ)
	$(CC) $(FLAGS) $@.c -o $(BIN)/$@ $(OBJ) $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

superpixel: $(libift) $(OBJ)
	$(MAKE) iftAggregateCSVStats
	$(MAKE) iftComputeSuperpixelMetrics
	$(MAKE) iftFixLabelMap
	$(MAKE) iftHISF_segmentation
	$(MAKE) iftInterpolateSampleStats
	$(MAKE) iftOverlayBorders
	$(MAKE) iftSortCSV

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o
	rm -df $(ODIR)
	rm -f $(BIN)/*;

$(OBJ): | $(ODIR)

$(ODIR):
	mkdir $(ODIR)
