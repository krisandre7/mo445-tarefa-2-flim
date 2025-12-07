all: libmodulesliceview

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulesliceview: $(LIB)/libmodulesliceview.a
	echo "libmodulesliceview.a built..."

$(LIB)/libmodulesliceview.a: \
$(OBJ)/modulesliceview.o \
$(OBJ)/slicecanvas.o \
$(OBJ)/sliceview.o \
$(OBJ)/sliceviewopt.o
	ar csr $(LIB)/libmodulesliceview.a \
$(OBJ)/modulesliceview.o \
$(OBJ)/slicecanvas.o \
$(OBJ)/sliceview.o \
$(OBJ)/sliceviewopt.o


$(OBJ)/sliceviewopt.o: $(SRC)/sliceviewopt.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/sliceviewopt.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/sliceviewopt.o

$(OBJ)/slicecanvas.o: $(SRC)/slicecanvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/slicecanvas.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/slicecanvas.o

$(OBJ)/sliceview.o: $(SRC)/sliceview.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/sliceview.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/sliceview.o

$(OBJ)/modulesliceview.o: $(SRC)/modulesliceview.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulesliceview.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulesliceview.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



