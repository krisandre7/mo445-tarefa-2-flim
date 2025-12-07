
BIN=../../bin

#FLAGS= -g -O0 -Wall -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -O3 -Wall -fopenmp -pthread -std=gnu11 -pedantic
IFTLIB    = -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lm
LIBSVM    = -L$(NEWIFT_DIR)/libsvm -lsvm -lstdc++ 
IFTFLAGS  = -I$(NEWIFT_DIR)/include 

libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\

libsvm:
	cd ../../libsvm; $(MAKE) -C $(NEWIFT_DIR)/libsvm; cd -\

msconvnet: libift libsvm msconvnet.c
	make 
	gcc $(FLAGS) msconvnet.c -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \


$@.c: libift $@.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB); \

clean:
	rm $(BIN)/*;
