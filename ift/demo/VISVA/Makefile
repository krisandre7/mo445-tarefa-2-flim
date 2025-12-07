# VISVA - Volumetric Image Segmentation for Visualization and Analysis
# For windows, type: make visva-win
# Set CC_FLAGS environment variable to "-Wall -lpthread -msse -march=native -lm -lz"
# Static libraries: http://www.yolinux.com/TUTORIALS/LibraryArchives-StaticAndDynamic.html

all: visva
	@echo "DONE"

#Compiler
export CXX=g++ 
export CC=gcc
export CC_FLAGS = -O3 -Wall -fopenmp -msse -march=native -w -lm -lz -g 
#export CC_FLAGS = -ggdb -O0 -Wall -lpthread -fopenmp -march=native -msse -lz -lm -llapack -lblas -w -fsanitize=address

#Directories
export VISVA_DIR = $(NEWIFT_DIR)/demo/VISVA
export CODE_C_DIR = $(VISVA_DIR)/externals/code_c
export CODE_CPP_DIR = $(VISVA_DIR)/externals/code_cpp
export GUI_DIR = $(VISVA_DIR)/externals/gui
export OPF_DIR = $(VISVA_DIR)/externals/LibOPF
export OLDIFT_DIR = $(VISVA_DIR)/externals/liboldift

###########  Libraries ##############
export WXLIBS       =`wx-config --libs`
export CODE_C_LIB   = -L$(CODE_C_DIR)/lib -lcode_c
export CODE_CPP_LIB = -L$(CODE_CPP_DIR)/lib -lcode_cpp
export GUI_LIB      = -L$(GUI_DIR) -lgui
export OPF_LIB      = -L$(OPF_DIR)/lib -lOPF
export OLDIFT_LIB   = -L$(OLDIFT_DIR)/lib -loldift -lz 
export NEWIFT_LIB   = -L$(NEWIFT_DIR)/lib -L/usr/lib/atlas-base -L/usr/lib/atlas-base/atlas -lift -lcblas -lblas -llapack
export LIBSVM       = -L$(NEWIFT_DIR)/externals/libsvm/lib -lsvm -lstdc++
# export MERGED_LIB	  = -L$(VISVA_DIR)/lib -lmerged

###########   Includes ##############
export WXFLAGS=`wx-config --cflags`
export VISVA_INCLUDE    = -I$(VISVA_DIR)
export NEWIFT_INCLUDE   = -I$(NEWIFT_DIR)/include -I/System/Library/Frameworks/vecLib.framework/Headers/
export CODE_C_INCLUDE   = -I$(CODE_C_DIR)/include -I$(NEWIFT_DIR)/externals/libsvm/include 
export CODE_CPP_INCLUDE = -I$(CODE_CPP_DIR)/include 
export GUI_INCLUDE      = -I$(GUI_DIR)/include 
export OPF_INCLUDE      = -I$(OPF_DIR)/include
export OLDIFT_INCLUDE      = -I$(OLDIFT_DIR)/include

###########   Modules ###########
export MODULE_DIR = $(VISVA_DIR)/modules
export BASIC_DIR = $(MODULE_DIR)/basic
export BASIC_INCLUDE = -I$(BASIC_DIR)/include

export SURFACE_DIR = $(MODULE_DIR)/visualization/surface
export SURFACE_INCLUDE = -I$(SURFACE_DIR)/include

export MODULE_LIB  = -L$(MODULE_DIR) -lmodules
export MODULE_INCLUDE = $(SURFACE_INCLUDE) $(BASIC_INCLUDE) -I$(MODULE_DIR)

#Rules
libift:
	$(MAKE) -C $(NEWIFT_DIR) 

liboldift:
	$(MAKE) -C $(OLDIFT_DIR)

libOPF: liboldift
	$(MAKE) -C $(OPF_DIR)

libcode_c:
	$(MAKE) -C $(CODE_C_DIR)

libcode_cpp:
	$(MAKE) -C $(CODE_CPP_DIR)

libgui:
	$(MAKE) -C $(GUI_DIR)

libmodules:
	$(MAKE) -C $(MODULE_DIR)


#http://www.oracle.com/technetwork/articles/servers-storage-dev/linkinglibraries4-405107.html
visva: visva.cpp libift liboldift libOPF  libcode_c libcode_cpp libgui libmodules 

	$(CXX) $(CC_FLAGS) $(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(OLDIFT_INCLUDE) \
	$(OPF_INCLUDE) $(WXFLAGS) $(GUI_INCLUDE) \
	$(MODULE_INCLUDE) $(VISVA_INCLUDE) $(NEWIFT_INCLUDE) \
	visva.cpp $(NEWIFT_LIB) $(OPF_LIB) $(OLDIFT_LIB) $(MODULE_LIB) $(GUI_LIB) $(WXLIBS) \
	$(CODE_CPP_LIB) $(CODE_C_LIB) $(OPF_LIB) $(OLDIFT_LIB) $(NEWIFT_LIB) $(LIBSVM) -o $(NEWIFT_DIR)/bin/visva
	chmod +x $(NEWIFT_DIR)/bin/visva

dicom2scn: dicom2scn.c
	$(CC) -Wall -O2 dicom2scn.c -o dicom2scn -lm

clean:
	$(RM) *~ *.o visva dicom2scn tmp.scn windows-icon.o -f
	$(MAKE) clean -C $(CODE_C_DIR)
	$(MAKE) clean -C $(CODE_CPP_DIR)
	$(MAKE) clean -C $(GUI_DIR)
	$(MAKE) clean -C $(MODULE_DIR)
	$(MAKE) clean -C $(NEWIFT_DIR)
	$(MAKE) clean -C $(OPF_DIR)
	$(MAKE) clean -C $(OLDIFT_DIR)
	rm -rf ./bin/*
	rm -rf ./modules/segmentation/cloudsbrain/runexp
	rm -rf ./modules/segmentation/cloudsbrain/runexp_1out

visva-win: visva.cpp libcode_c libcode_cpp libgui libmodules windows-icon.rc windows-icon.ico
	windres windows-icon.rc -O coff -o windows-icon.o
	$(CXX) $(CC_FLAGS) $(CODE_C_INCLUDE) $(IFT_INCLUDE) $(OPF_INCLUDE) \
	$(CODE_CPP_INCLUDE) $(WXFLAGS) $(GUI_INCLUDE) \
	$(MODULE_INCLUDE) $(VISVA_INCLUDE) \
	visva.cpp $(OLDIFT_LIB) $(MODULE_LIB) $(GUI_LIB) $(WXLIBS) \
	$(CODE_CPP_LIB) $(CODE_C_LIB) $(OPF_LIB) $(PTHREAD_FLAGS) \
	-o visva windows-icon.o
	@echo "DONE"

