all: libmodulecombine

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulecombine: $(LIB)/libmodulecombine.a
	echo "libmodulecombine.a built..."

$(LIB)/libmodulecombine.a: \
$(OBJ)/cpainthandler.o \
$(OBJ)/combinedialog.o \
$(OBJ)/modulecombine.o
	ar csr $(LIB)/libmodulecombine.a \
$(OBJ)/cpainthandler.o \
$(OBJ)/combinedialog.o \
$(OBJ)/modulecombine.o


$(OBJ)/cpainthandler.o: $(SRC)/cpainthandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/cpainthandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/cpainthandler.o

$(OBJ)/combinedialog.o: $(SRC)/combinedialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/combinedialog.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/combinedialog.o

$(OBJ)/modulecombine.o: $(SRC)/modulecombine.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulecombine.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulecombine.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



