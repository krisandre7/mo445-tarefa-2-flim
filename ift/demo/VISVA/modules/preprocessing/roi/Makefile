all: libmoduleroi

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmoduleroi: $(LIB)/libmoduleroi.a
	echo "libmoduleroi.a built..."

$(LIB)/libmoduleroi.a: \
$(OBJ)/roidialog.o \
$(OBJ)/moduleroi.o
	ar csr $(LIB)/libmoduleroi.a \
$(OBJ)/roidialog.o \
$(OBJ)/moduleroi.o


$(OBJ)/roidialog.o: $(SRC)/roidialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/roidialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/roidialog.o

$(OBJ)/moduleroi.o: $(SRC)/moduleroi.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleroi.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/moduleroi.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



