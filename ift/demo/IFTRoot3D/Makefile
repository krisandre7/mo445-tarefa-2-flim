BIN=$(NEWIFT_DIR)/bin
SRC=$(NEWIFT_DIR)/demo/IFTRoot3D/src
LIB=$(NEWIFT_DIR)/lib
OBJ=$(NEWIFT_DIR)/obj
INC=$(NEWIFT_DIR)/include

#FLAGS= -g -O0 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -O3 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
IFTLIB= -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lm -L/lib64/atlas-sse3/ -L/usr/lib/atlas-base -lcblas -lpng -lz
LIBSVM= -L$(NEWIFT_DIR)/libsvm -lsvm -lstdc++ 
INCLUDES= -I$(NEWIFT_DIR)/include 

DEBUG=0
ifeq ($(DEBUG), 1)
	FLAGS += -g
endif

all: 
#	cd $(NEWIFT_DIR); $(MAKE) clean; $(MAKE) -C $(NEWIFT_DIR); cd - ;\

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftSelectTraitsByVPS.c $(SRC)/iftRoot3D.c -o $(BIN)/iftSelectTraitsByVPS $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftSelectTraitsByMSPS.c $(SRC)/iftRoot3D.c -o $(BIN)/iftSelectTraitsByMSPS $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftCreatePhenotypeFile.c $(SRC)/iftRoot3D.c -o $(BIN)/iftCreatePhenotypeFile $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftGroupTraits.c $(SRC)/iftRoot3D.c -o $(BIN)/iftGroupTraits $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftEliminateSmallGroups.c $(SRC)/iftRoot3D.c -o $(BIN)/iftEliminateSmallGroups $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftSelectSubpop.c $(SRC)/iftRoot3D.c -o $(BIN)/iftSelectSubpop $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftGWAA.c $(SRC)/iftRoot3D.c -o $(BIN)/iftGWAA $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftGenoPlots.c -o $(BIN)/iftGenoPlots $(IFTFLAGS) $(IFTLIB) ; \

#	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftStatTable.c -o $(BIN)/iftStatTable $(IFTFLAGS) $(IFTLIB) ; \

	gcc $(FLAGS) $(INCLUDES) $(SRC)/iftDetectRootMesh.c -o $(BIN)/iftDetectRootMesh $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

clean:
	rm -f $(BIN)/* $(LIB)/libift.a $(OBJ)/*.o




