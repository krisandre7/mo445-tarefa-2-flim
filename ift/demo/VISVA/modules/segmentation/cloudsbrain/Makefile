all: libmodule_cloudsbrain standalone_cloudsbrain runexp runexp_1out

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmodule_cloudsbrain: $(LIB)/libmodulecloudsbrain.a
	echo "libmodulecloudsbrain.a built..."

$(LIB)/libmodulecloudsbrain.a: \
$(OBJ)/methods_addons.o \
$(OBJ)/gradient_addons.o \
$(OBJ)/cloudsbrain.o \
$(OBJ)/modulecloudsbrain.o
	ar csr $(LIB)/libmodulecloudsbrain.a \
$(OBJ)/methods_addons.o \
$(OBJ)/gradient_addons.o \
$(OBJ)/cloudsbrain.o \
$(OBJ)/modulecloudsbrain.o


$(OBJ)/methods_addons.o: $(SRC)/methods_addons.cpp $(HDR)/methods_addons.h
	$(CXX)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	-c $(SRC)/methods_addons.cpp -I$(HDR) \
	$(CC_FLAGS) -o $(OBJ)/methods_addons.o

$(OBJ)/gradient_addons.o: $(SRC)/gradient_addons.c $(HDR)/gradient_addons.h
	$(CC)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_C_INCLUDE) \
	-c $(SRC)/gradient_addons.c -I$(HDR) \
	$(CC_FLAGS) -o $(OBJ)/gradient_addons.o

#$(OBJ)/cloud3_addons.o: $(SRC)/cloud3_addons.c $(HDR)/cloud3_addons.h
#	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_C_INCLUDE) \
#	-c $(SRC)/cloud3_addons.c -I$(HDR) \
#	-o $(OBJ)/cloud3_addons.o


$(OBJ)/modulecloudsbrain.o: $(SRC)/modulecloudsbrain.cpp
	$(CXX)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulecloudsbrain.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/modulecloudsbrain.o

$(OBJ)/cloudsbrain.o: $(SRC)/cloudsbrain.cpp
	$(CXX)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/cloudsbrain.cpp -I$(HDR) \
	$(WXFLAGS) $(CC_FLAGS) -o $(OBJ)/cloudsbrain.o


standalone_cloudsbrain: $(SRC)/cloudsbrain.cpp 
	$(CXX)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) $(WXFLAGS) \
	-D_CLOUDSBRAIN_STANDALONE_ \
	$(SRC)/cloudsbrain.cpp -I$(HDR) \
	$(OBJ)/methods_addons.o	\
	$(OBJ)/gradient_addons.o \
	$(CODE_C_LIB) $(CODE_CPP_LIB) \
	$(OPF_LIB) $(OLDIFT_LIB) -lm -lpthread \
	$(WXLIBS) $(CC_FLAGS) -o $(VISVA_DIR)/bin/cloudsbrain_bin

runexp: runexp.c
	$(CC)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_C_INCLUDE) \
	runexp.c $(CODE_C_LIB) \
	$(OLDIFT_LIB) $(OPF_LIB) $(CC_FLAGS) -o runexp -lm -lpthread

runexp_1out: runexp_1out.c
	$(CC)  $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_C_INCLUDE) \
	runexp_1out.c $(CODE_C_LIB) \
	$(OLDIFT_LIB) $(OPF_LIB) $(CC_FLAGS) -o runexp_1out -lm -lpthread

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ runexp runexp_1out $(VISVA_DIR)/bin/cloudsbrain_bin -f



