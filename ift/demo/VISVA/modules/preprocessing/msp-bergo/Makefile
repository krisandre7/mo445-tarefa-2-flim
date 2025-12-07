all: libmodulemsp

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulemsp: $(LIB)/libmodulemsp.a
	echo "libmodulemsp.a built..."

$(LIB)/libmodulemsp.a: \
$(OBJ)/msp_dialog.o \
$(OBJ)/msp_libalign.o \
$(OBJ)/msp_align.o \
$(OBJ)/modulemsp.o
	ar csr $(LIB)/libmodulemsp.a \
$(OBJ)/msp_dialog.o \
$(OBJ)/msp_libalign.o \
$(OBJ)/msp_align.o \
$(OBJ)/modulemsp.o


$(OBJ)/msp_libalign.o: $(SRC)/msp_libalign.cc
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/msp_libalign.cc -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/msp_libalign.o

$(OBJ)/msp_align.o: $(SRC)/msp_align.cc
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/msp_align.cc -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/msp_align.o

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

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



