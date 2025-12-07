all: libmoduleenhancement

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmoduleenhancement: $(LIB)/libmoduleenhancement.a
	echo "libmoduleenhancement.a built..."

$(LIB)/libmoduleenhancement.a: \
$(OBJ)/delsamples.o \
$(OBJ)/addsamples.o \
$(OBJ)/enhancementdialog.o \
$(OBJ)/moduleenhancement.o
	ar csr $(LIB)/libmoduleenhancement.a \
$(OBJ)/delsamples.o \
$(OBJ)/addsamples.o \
$(OBJ)/enhancementdialog.o \
$(OBJ)/moduleenhancement.o


$(OBJ)/addsamples.o: $(SRC)/addsamples.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/addsamples.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/addsamples.o

$(OBJ)/delsamples.o: $(SRC)/delsamples.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/delsamples.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/delsamples.o

$(OBJ)/enhancementdialog.o: $(SRC)/enhancementdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/enhancementdialog.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/enhancementdialog.o

$(OBJ)/moduleenhancement.o: $(SRC)/moduleenhancement.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleenhancement.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/moduleenhancement.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



