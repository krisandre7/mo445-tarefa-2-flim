all: libmoduleselfreg

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmoduleselfreg: $(LIB)/libmoduleselfreg.a
	echo "libmoduleselfreg.a built..."

$(LIB)/libmoduleselfreg.a: \
$(OBJ)/selfreg_dialog.o \
$(OBJ)/moduleselfreg.o
	ar csr $(LIB)/libmoduleselfreg.a \
$(OBJ)/selfreg_dialog.o \
$(OBJ)/moduleselfreg.o


$(OBJ)/selfreg_dialog.o: $(SRC)/selfreg_dialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/selfreg_dialog.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/selfreg_dialog.o


$(OBJ)/moduleselfreg.o: $(SRC)/moduleselfreg.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleselfreg.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduleselfreg.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



