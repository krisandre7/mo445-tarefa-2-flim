all: libmoduleinterpolation

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmoduleinterpolation: $(LIB)/libmoduleinterpolation.a
	echo "libmoduleinterpolation.a built..."

$(LIB)/libmoduleinterpolation.a: \
$(OBJ)/interpolationdialog.o \
$(OBJ)/moduleinterpolation.o
	ar csr $(LIB)/libmoduleinterpolation.a \
$(OBJ)/interpolationdialog.o \
$(OBJ)/moduleinterpolation.o


$(OBJ)/interpolationdialog.o: $(SRC)/interpolationdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/interpolationdialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/interpolationdialog.o

$(OBJ)/moduleinterpolation.o: $(SRC)/moduleinterpolation.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleinterpolation.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/moduleinterpolation.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



