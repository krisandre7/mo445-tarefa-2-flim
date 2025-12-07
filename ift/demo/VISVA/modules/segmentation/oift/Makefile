all: libmoduleoift

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmoduleoift: $(LIB)/libmoduleoift.a
	echo "libmoduleoift.a built..."

$(LIB)/libmoduleoift.a: \
$(OBJ)/moduleoift.o \
$(OBJ)/addhandleroift.o \
$(OBJ)/delhandleroift.o \
$(OBJ)/livehandleroift.o \
$(OBJ)/oiftdialog.o \
$(OBJ)/livewireoift.o
	ar csr $(LIB)/libmoduleoift.a \
$(OBJ)/delhandleroift.o \
$(OBJ)/addhandleroift.o \
$(OBJ)/moduleoift.o \
$(OBJ)/oiftdialog.o \
$(OBJ)/livehandleroift.o \
$(OBJ)/livewireoift.o


$(OBJ)/addhandleroift.o: $(SRC)/addhandleroift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/addhandleroift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB) -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/addhandleroift.o

$(OBJ)/delhandleroift.o: $(SRC)/delhandleroift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/delhandleroift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/delhandleroift.o

$(OBJ)/livehandleroift.o: $(SRC)/livehandleroift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/livehandleroift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/livehandleroift.o

$(OBJ)/oiftdialog.o: $(SRC)/oiftdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/oiftdialog.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/oiftdialog.o

$(OBJ)/moduleoift.o: $(SRC)/moduleoift.cpp
	$(CXX) $(CC_FLAGS) $(NEWIFT_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleoift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduleoift.o

$(OBJ)/livewireoift.o: $(SRC)/livewireoift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/livewireoift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	-o $(OBJ)/livewireoift.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



