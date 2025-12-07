all: libmodsegexternal

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmodsegexternal: $(LIB)/libmodsegexternal.a
	echo "libmodsegexternal.a built..."

$(LIB)/libmodsegexternal.a: \
$(OBJ)/modsegexternal.o
	ar csr $(LIB)/libmodsegexternal.a \
$(OBJ)/modsegexternal.o


$(OBJ)/modsegexternal.o: $(SRC)/modsegexternal.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modsegexternal.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modsegexternal.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



