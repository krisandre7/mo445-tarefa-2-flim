all: libmodulemanual

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmodulemanual: $(LIB)/libmodulemanual.a
	echo "libmodulemanual.a built..."

$(LIB)/libmodulemanual.a: \
$(OBJ)/painthandler.o \
$(OBJ)/manualopt.o \
$(OBJ)/manualdialog.o \
$(OBJ)/modulemanual.o
	ar csr $(LIB)/libmodulemanual.a \
$(OBJ)/painthandler.o \
$(OBJ)/manualopt.o \
$(OBJ)/manualdialog.o \
$(OBJ)/modulemanual.o


$(OBJ)/painthandler.o: $(SRC)/painthandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/painthandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/painthandler.o

$(OBJ)/manualopt.o: $(SRC)/manualopt.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/manualopt.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/manualopt.o

$(OBJ)/manualdialog.o: $(SRC)/manualdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/manualdialog.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/manualdialog.o

$(OBJ)/modulemanual.o: $(SRC)/modulemanual.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulemanual.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulemanual.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



