all: libmodulebraincluster standalone_braincluster standalone_atlas_braincluster

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmodulebraincluster: $(LIB)/libmodulebraincluster.a
	echo "libmodulebraincluster.a built..."

$(LIB)/libmodulebraincluster.a: \
$(OBJ)/braincluster_dialog.o \
$(OBJ)/modulebraincluster.o
	ar csr $(LIB)/libmodulebraincluster.a \
$(OBJ)/braincluster_dialog.o \
$(OBJ)/modulebraincluster.o


$(OBJ)/braincluster_dialog.o: $(SRC)/braincluster_dialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/braincluster_dialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/braincluster_dialog.o

$(OBJ)/modulebraincluster.o: $(SRC)/modulebraincluster.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulebraincluster.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/modulebraincluster.o

standalone_braincluster: 
	$(CC)  $(CODE_C_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
	-D_BRAINCLUSTER_STANDALONE_ \
	$(CODE_C_DIR)/src/braincluster_spectral.c -I$(HDR) \
	$(CODE_C_LIB) $(OPF_LIB) $(OLDIFT_LIB) \
	$(CC_FLAGS) -o $(VISVA_DIR)/bin/braincluster_spectral_bin

standalone_atlas_braincluster: 
	$(CC)  $(CODE_C_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
	-D_BRAINCLUSTER_ATLAS_STANDALONE_ \
	$(CODE_C_DIR)/src/braincluster_atlas.c -I$(HDR) \
	$(CODE_C_LIB) $(OPF_LIB) $(OLDIFT_LIB) \
	$(CC_FLAGS) -o $(VISVA_DIR)/bin/braincluster_atlas_bin

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ $(VISVA_DIR)/bin/braincluster_*_bin -f



