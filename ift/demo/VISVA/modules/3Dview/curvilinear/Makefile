all: libmodulecurvilinear

LIB=.
HDR=./include
SRC=./src
OBJ=./obj

#Rules
libmodulecurvilinear: $(LIB)/libmodulecurvilinear.a
	echo "libmodulecurvilinear.a built..."

$(LIB)/libmodulecurvilinear.a: \
$(OBJ)/modulecurvilinear.o \
$(OBJ)/curvilinearcanvas.o \
$(OBJ)/curvilinearview.o \
$(OBJ)/curvilinearhandler.o
	ar csr $(LIB)/libmodulecurvilinear.a \
$(OBJ)/modulecurvilinear.o \
$(OBJ)/curvilinearcanvas.o \
$(OBJ)/curvilinearview.o \
$(OBJ)/curvilinearhandler.o

$(OBJ)/curvilinearhandler.o: $(SRC)/curvilinearhandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/curvilinearhandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/curvilinearhandler.o

$(OBJ)/curvilinearcanvas.o: $(SRC)/curvilinearcanvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/curvilinearcanvas.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/curvilinearcanvas.o

$(OBJ)/curvilinearview.o: $(SRC)/curvilinearview.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/curvilinearview.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/curvilinearview.o

$(OBJ)/modulecurvilinear.o: $(SRC)/modulecurvilinear.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulecurvilinear.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulecurvilinear.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



