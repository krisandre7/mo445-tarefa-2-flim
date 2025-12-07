all: libmodulecsf

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulecsf: $(LIB)/libmodulecsf.a
	echo "libmodulecsf.a built..."

$(LIB)/libmodulecsf.a: \
$(OBJ)/modulecsf.o \
$(OBJ)/csf.o
	ar csr $(LIB)/libmodulecsf.a \
$(OBJ)/modulecsf.o \
$(OBJ)/csf.o


$(OBJ)/modulecsf.o: $(SRC)/modulecsf.cpp 
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulecsf.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulecsf.o

$(OBJ)/csf.o: $(SRC)/csf.c 
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
	-c $(SRC)/csf.c -I$(HDR) \
	-o $(OBJ)/csf.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



