all: libmodulemsp standalone_msp

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulemsp: $(LIB)/libmodulemsp.a
	echo "libmodulemsp.a built..."

$(LIB)/libmodulemsp.a: \
$(OBJ)/msp_dialog.o \
$(OBJ)/modulemsp.o
	ar csr $(LIB)/libmodulemsp.a \
$(OBJ)/msp_dialog.o \
$(OBJ)/modulemsp.o


$(OBJ)/msp_dialog.o: $(SRC)/msp_dialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/msp_dialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/msp_dialog.o

$(OBJ)/modulemsp.o: $(SRC)/modulemsp.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulemsp.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/modulemsp.o


standalone_msp: 
	$(CC)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_CPP_INCLUDE) \
	-D_MSP_STANDALONE_ \
	$(SRC)/msp_bin.c -I$(HDR) \
	$(CODE_C_LIB) $(OLDIFT_LIB) $(OPF_LIB) \
	$(CC_FLAGS) -o $(VISVA_DIR)/bin/msp_bin


clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f $(VISVA_DIR)/bin/msp_bin



