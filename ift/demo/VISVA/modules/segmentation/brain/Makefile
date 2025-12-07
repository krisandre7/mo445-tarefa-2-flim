all: libmodulebrain

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulebrain: $(LIB)/libmodulebrain.a
	echo "libmodulebrain.a built..."

$(LIB)/libmodulebrain.a: \
$(OBJ)/modulebrain.o
	ar csr $(LIB)/libmodulebrain.a \
$(OBJ)/modulebrain.o


$(OBJ)/modulebrain.o: $(SRC)/modulebrain.cpp 
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulebrain.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/modulebrain.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



