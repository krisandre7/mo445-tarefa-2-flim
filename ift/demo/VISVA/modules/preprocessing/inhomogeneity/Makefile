all: libmoduleinhomogeneity standalone_inhomogeneity_brain

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmoduleinhomogeneity: $(LIB)/libmoduleinhomogeneity.a
	echo "libmoduleinhomogeneity.a built..."

$(LIB)/libmoduleinhomogeneity.a: \
$(OBJ)/inhomogeneity_dialog.o \
$(OBJ)/moduleinhomogeneity.o
	ar csr $(LIB)/libmoduleinhomogeneity.a \
$(OBJ)/inhomogeneity_dialog.o \
$(OBJ)/moduleinhomogeneity.o


$(OBJ)/inhomogeneity_dialog.o: $(SRC)/inhomogeneity_dialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/inhomogeneity_dialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/inhomogeneity_dialog.o

$(OBJ)/moduleinhomogeneity.o: $(SRC)/moduleinhomogeneity.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleinhomogeneity.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/moduleinhomogeneity.o

standalone_inhomogeneity_brain: 
	$(CC)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_C_INCLUDE) \
	-D_INHOMOGENEITY_BRAIN_STANDALONE_ \
	$(CODE_C_DIR)/src/inhomogeneity.c -I$(HDR) \
	$(CODE_C_LIB) $(OPF_LIB) $(OLDIFT_LIB) \
	$(CC_FLAGS) -o $(VISVA_DIR)/bin/inhomogeneity_brain_bin

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~  $(VISVA_DIR)/bin/inhomogeneity_brain_bin -f



