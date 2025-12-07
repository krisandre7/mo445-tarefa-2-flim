all: libmodulesymmetry

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulesymmetry: $(LIB)/libmodulesymmetry.a
	echo "libmodulesymmetry.a built..."

$(LIB)/libmodulesymmetry.a: \
$(OBJ)/modulesymmetry.o
	ar csr $(LIB)/libmodulesymmetry.a \
$(OBJ)/modulesymmetry.o


$(OBJ)/modulesymmetry.o: $(SRC)/modulesymmetry.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulesymmetry.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulesymmetry.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



