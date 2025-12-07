all: libmodulerigidreg

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulerigidreg: $(LIB)/libmodulerigidreg.a
	echo "libmodulerigidreg.a built..."

$(LIB)/libmodulerigidreg.a: \
$(OBJ)/rigidreg_dialog.o \
$(OBJ)/modulerigidreg.o
	ar csr $(LIB)/libmodulerigidreg.a \
$(OBJ)/rigidreg_dialog.o \
$(OBJ)/modulerigidreg.o


$(OBJ)/rigidreg_dialog.o: $(SRC)/rigidreg_dialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/rigidreg_dialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/rigidreg_dialog.o

$(OBJ)/modulerigidreg.o: $(SRC)/modulerigidreg.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulerigidreg.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/modulerigidreg.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



