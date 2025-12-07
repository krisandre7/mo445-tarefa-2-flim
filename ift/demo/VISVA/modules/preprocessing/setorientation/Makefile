all: libmodulesetorientation standalone_setorientation_brain

LIB=.
HDR=./include
SRC=./src
OBJ=./obj

#Rules
libmodulesetorientation: $(LIB)/libmodulesetorientation.a
	echo "libmodulesetorientation.a built..."

$(LIB)/libmodulesetorientation.a: \
$(OBJ)/setorientationdialog.o \
$(OBJ)/modulesetorientation.o
	ar csr $(LIB)/libmodulesetorientation.a \
$(OBJ)/setorientationdialog.o \
$(OBJ)/modulesetorientation.o


$(OBJ)/setorientationdialog.o: $(SRC)/setorientationdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/setorientationdialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/setorientationdialog.o

$(OBJ)/modulesetorientation.o: $(SRC)/modulesetorientation.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulesetorientation.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/modulesetorientation.o

standalone_setorientation_brain:
	$(CC)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE)  $(CODE_C_INCLUDE) \
	$(SRC)/setorientation.c $(CODE_C_LIB) $(OPF_LIB) $(OLDIFT_LIB)\
	$(CC_FLAGS) -o $(VISVA_DIR)/bin/setorientation_brain_bin

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



