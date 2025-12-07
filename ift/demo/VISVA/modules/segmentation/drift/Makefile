all: libmoduledrift

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmoduledrift: $(LIB)/libmoduledrift.a
	echo "libmoduledrift.a built..."

$(LIB)/libmoduledrift.a: \
$(OBJ)/moduledrift.o \
$(OBJ)/addhandlerdrift.o \
$(OBJ)/delhandlerdrift.o \
$(OBJ)/livehandlerdrift.o \
$(OBJ)/driftdialog.o \
$(OBJ)/livewiredrift.o 
	ar csr $(LIB)/libmoduledrift.a \
$(OBJ)/delhandlerdrift.o \
$(OBJ)/addhandlerdrift.o \
$(OBJ)/moduledrift.o \
$(OBJ)/driftdialog.o \
$(OBJ)/livehandlerdrift.o \
$(OBJ)/livewiredrift.o


$(OBJ)/addhandlerdrift.o: $(SRC)/addhandlerdrift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/addhandlerdrift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB) -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/addhandlerdrift.o

$(OBJ)/delhandlerdrift.o: $(SRC)/delhandlerdrift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/delhandlerdrift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/delhandlerdrift.o

$(OBJ)/livehandlerdrift.o: $(SRC)/livehandlerdrift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/livehandlerdrift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/livehandlerdrift.o

$(OBJ)/driftdialog.o: $(SRC)/driftdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/driftdialog.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/driftdialog.o

$(OBJ)/moduledrift.o: $(SRC)/moduledrift.cpp
	$(CXX) $(CC_FLAGS) $(NEWIFT_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduledrift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduledrift.o

$(OBJ)/livewiredrift.o: $(SRC)/livewiredrift.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/livewiredrift.cpp $(NEWIFT_LIB) $(OLDIFT_LIB)-I$(HDR) \
	-o $(OBJ)/livewiredrift.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



