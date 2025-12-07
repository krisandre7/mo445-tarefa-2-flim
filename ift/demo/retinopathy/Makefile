BIN=../../bin

#FLAGS= -g -O0 -Wall -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -O3 -fopenmp -pthread -std=gnu11 -pedantic
IFTLIB     = -L$(NEWIFT_DIR)/lib -lift -ltiff -llapack -lblas -lm

# for static linking
#IFTLIB     = -ltiff -ljbig2dec -ljpeg -lz -llzma -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lcblas -lgfortran -lm
#FLAGS= -O3 -fopenmp -pthread -std=gnu11 -pedantic -static -Wl,--allow-multiple-definition -Wl,--unresolved-symbols=ignore-in-object-files
## --allow-multiple-definition ==> due to lapack & blas redefinition for static linking
## --warn-unresolved-symbols ==> problems with jbig2 static library


LIBSVM    = -L$(NEWIFT_DIR)/libsvm -lsvm -lm -lstdc++ 
INCLUDES  = -I$(NEWIFT_DIR)/include -I/usr/local/include
#INCLUDES  = -I$(NEWIFT_DIR)/include 


libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\

libsvm:
	cd ../../libsvm; $(MAKE) -C .; cd -\


$@.c: $@.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(INCLUDES) $(IFTLIB) $(LIBSVM); \

clean:
	rm $(BIN)/*;

