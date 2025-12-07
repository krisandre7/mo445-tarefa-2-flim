all: libmoduleinteractive

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmoduleinteractive: $(LIB)/libmoduleinteractive.a
	echo "libmoduleinteractive.a built..."

$(LIB)/libmoduleinteractive.a: \
$(OBJ)/delhandler.o \
$(OBJ)/addhandler.o \
$(OBJ)/interactiveopt.o \
$(OBJ)/interactivedialog.o \
$(OBJ)/moduleinteractive.o \
$(OBJ)/livehandler.o \
$(OBJ)/livewire.o
	ar csr $(LIB)/libmoduleinteractive.a \
$(OBJ)/delhandler.o \
$(OBJ)/addhandler.o \
$(OBJ)/interactiveopt.o \
$(OBJ)/interactivedialog.o \
$(OBJ)/moduleinteractive.o \
$(OBJ)/livehandler.o \
$(OBJ)/livewire.o


$(OBJ)/addhandler.o: $(SRC)/addhandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/addhandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/addhandler.o

$(OBJ)/delhandler.o: $(SRC)/delhandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/delhandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/delhandler.o

$(OBJ)/livehandler.o: $(SRC)/livehandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/livehandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/livehandler.o

$(OBJ)/interactiveopt.o: $(SRC)/interactiveopt.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/interactiveopt.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/interactiveopt.o

$(OBJ)/interactivedialog.o: $(SRC)/interactivedialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/interactivedialog.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/interactivedialog.o

$(OBJ)/moduleinteractive.o: $(SRC)/moduleinteractive.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleinteractive.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduleinteractive.o

$(OBJ)/livewire.o: $(SRC)/livewire.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/livewire.cpp -I$(HDR) \
	-o $(OBJ)/livewire.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



