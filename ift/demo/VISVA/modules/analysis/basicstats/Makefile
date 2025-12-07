all: libmodulebasicstats

LIB=.
HDR=./include
SRC=./src
OBJ=./obj



#Rules
libmodulebasicstats: $(LIB)/libmodulebasicstats.a
	echo "libmodulebasicstats.a built..."

$(LIB)/libmodulebasicstats.a: \
$(OBJ)/modulebasicstats.o
	ar csr $(LIB)/libmodulebasicstats.a \
$(OBJ)/modulebasicstats.o


$(OBJ)/modulebasicstats.o: $(SRC)/modulebasicstats.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(VISVA_INCLUDE) \
	$(CODE_C_INCLUDE) $(CODE_CPP_INCLUDE) \
	$(GUI_INCLUDE) $(MODULE_INCLUDE) \
	-c $(SRC)/modulebasicstats.cpp -I$(HDR) \
	$(WXFLAGS) -o $(OBJ)/modulebasicstats.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f



