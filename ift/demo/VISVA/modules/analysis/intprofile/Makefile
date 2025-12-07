all: libmoduleintprofile

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmoduleintprofile: $(LIB)/libmoduleintprofile.a
	echo "libmoduleintprofile.a built..."

$(LIB)/libmoduleintprofile.a: \
$(OBJ)/linehandler.o \
$(OBJ)/poshandler.o \
$(OBJ)/profiledialog.o \
$(OBJ)/moduleintprofile.o
	ar csr $(LIB)/libmoduleintprofile.a \
$(OBJ)/linehandler.o \
$(OBJ)/poshandler.o \
$(OBJ)/profiledialog.o \
$(OBJ)/moduleintprofile.o


$(OBJ)/linehandler.o: $(SRC)/linehandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/linehandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/linehandler.o

$(OBJ)/poshandler.o: $(SRC)/poshandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/poshandler.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/poshandler.o

$(OBJ)/profiledialog.o: $(SRC)/profiledialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/profiledialog.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/profiledialog.o

$(OBJ)/moduleintprofile.o: $(SRC)/moduleintprofile.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/moduleintprofile.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/moduleintprofile.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



