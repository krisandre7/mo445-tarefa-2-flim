all: libmodulerayleaping

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulerayleaping: $(LIB)/libmodulerayleaping.a
	echo "libmodulerayleaping.a built..."

$(LIB)/libmodulerayleaping.a: \
$(OBJ)/modulerayleaping.o \
$(OBJ)/rayleapingcanvas.o \
$(OBJ)/rayleapingview.o \
$(OBJ)/rayleapinghandler.o
	ar csr $(LIB)/libmodulerayleaping.a \
$(OBJ)/modulerayleaping.o \
$(OBJ)/rayleapingcanvas.o \
$(OBJ)/rayleapingview.o \
$(OBJ)/rayleapinghandler.o

$(OBJ)/rayleapinghandler.o: $(SRC)/rayleapinghandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/rayleapinghandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/rayleapinghandler.o

$(OBJ)/rayleapingcanvas.o: $(SRC)/rayleapingcanvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/rayleapingcanvas.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/rayleapingcanvas.o

$(OBJ)/rayleapingview.o: $(SRC)/rayleapingview.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/rayleapingview.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/rayleapingview.o

$(OBJ)/modulerayleaping.o: $(SRC)/modulerayleaping.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulerayleaping.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulerayleaping.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



