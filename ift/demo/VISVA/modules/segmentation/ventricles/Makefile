all: libmoduleventr

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmoduleventr: $(LIB)/libmoduleventr.a
	echo "libmoduleventr.a built..."

$(LIB)/libmoduleventr.a: \
$(OBJ)/ventropt.o \
$(OBJ)/moduleventr.o
	ar csr $(LIB)/libmoduleventr.a \
$(OBJ)/ventropt.o \
$(OBJ)/moduleventr.o


$(OBJ)/ventropt.o: $(SRC)/ventropt.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/ventropt.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/ventropt.o

$(OBJ)/moduleventr.o: $(SRC)/moduleventr.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleventr.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduleventr.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



