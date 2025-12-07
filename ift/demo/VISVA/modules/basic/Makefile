all: libmodulebasic

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libmodulebasic: $(LIB)/libmodulebasic.a
	echo "libmodulebasic.a built..."

$(LIB)/libmodulebasic.a: \
$(OBJ)/module.o \
$(OBJ)/modulepanel.o \
$(OBJ)/moduleregistration.o \
$(OBJ)/addbasichandler.o \
$(OBJ)/delbasichandler.o \
$(OBJ)/navbasichandler.o
	ar csr $(LIB)/libmodulebasic.a \
$(OBJ)/module.o \
$(OBJ)/modulepanel.o \
$(OBJ)/moduleregistration.o \
$(OBJ)/addbasichandler.o \
$(OBJ)/delbasichandler.o \
$(OBJ)/navbasichandler.o


$(OBJ)/module.o: $(SRC)/module.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(GUI_INCLUDE) \
	-c $(SRC)/module.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/module.o

$(OBJ)/modulepanel.o: $(SRC)/modulepanel.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(GUI_INCLUDE) \
	-c $(SRC)/modulepanel.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulepanel.o

$(OBJ)/moduleregistration.o: $(SRC)/moduleregistration.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(GUI_INCLUDE) $(NEWIFT_INCLUDE) \
	-c $(SRC)/moduleregistration.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduleregistration.o

$(OBJ)/addbasichandler.o: $(SRC)/addbasichandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(GUI_INCLUDE) $(NEWIFT_INCLUDE) \
	-c $(SRC)/addbasichandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/addbasichandler.o

$(OBJ)/delbasichandler.o: $(SRC)/delbasichandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(GUI_INCLUDE) $(NEWIFT_INCLUDE) \
	-c $(SRC)/delbasichandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/delbasichandler.o

$(OBJ)/navbasichandler.o: $(SRC)/navbasichandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(GUI_INCLUDE) $(NEWIFT_INCLUDE) \
	-c $(SRC)/navbasichandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/navbasichandler.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



