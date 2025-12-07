all: libmodulethreshold

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmodulethreshold: $(LIB)/libmodulethreshold.a
	echo "libmodulethreshold.a built..."

$(LIB)/libmodulethreshold.a: \
$(OBJ)/thresholddialog.o \
$(OBJ)/modulethreshold.o
	ar csr $(LIB)/libmodulethreshold.a \
$(OBJ)/thresholddialog.o \
$(OBJ)/modulethreshold.o


$(OBJ)/thresholddialog.o: $(SRC)/thresholddialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/thresholddialog.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/thresholddialog.o

$(OBJ)/modulethreshold.o: $(SRC)/modulethreshold.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulethreshold.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulethreshold.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



