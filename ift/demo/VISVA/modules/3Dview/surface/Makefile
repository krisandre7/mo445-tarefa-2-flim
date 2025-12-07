all: libmodulesurface

LIB=.
HDR=./include
SRC=./src
OBJ=./obj

#Rules
libmodulesurface: $(LIB)/libmodulesurface.a
	echo "libmodulesurface.a built..."

$(LIB)/libmodulesurface.a: \
$(OBJ)/modulesurface.o \
$(OBJ)/surfacecanvas.o \
$(OBJ)/surfaceview.o \
$(OBJ)/surfacehandler.o
	ar csr $(LIB)/libmodulesurface.a \
$(OBJ)/modulesurface.o \
$(OBJ)/surfacecanvas.o \
$(OBJ)/surfaceview.o \
$(OBJ)/surfacehandler.o

$(OBJ)/surfacehandler.o: $(SRC)/surfacehandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/surfacehandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/surfacehandler.o

$(OBJ)/surfacecanvas.o: $(SRC)/surfacecanvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/surfacecanvas.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/surfacecanvas.o

$(OBJ)/surfaceview.o: $(SRC)/surfaceview.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/surfaceview.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/surfaceview.o

$(OBJ)/modulesurface.o: $(SRC)/modulesurface.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulesurface.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulesurface.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



