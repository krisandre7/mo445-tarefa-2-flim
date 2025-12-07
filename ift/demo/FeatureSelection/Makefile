BIN=../../bin

#FLAGS= -g -O0 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -O3 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
#IFTLIB    = -L$(NEWIFT_DIR)/lib  -lift -llapack -lblas -lm 
IFTLIB    = -L$(NEWIFT_DIR)/lib -L$(HOME)/atlas/lib -lift -llapack -lblas -latlas -lm
LIBSVM    = -L$(NEWIFT_DIR)/libsvm -lsvm -lstdc++ 
IFTFLAGS  = -I$(NEWIFT_DIR)/include -I$(HOME)/atlas/include
#IFTFLAGS  = -I$(NEWIFT_DIR)/include 

libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\

libsvm:
	cd ../../libsvm; $(MAKE) -C $(NEWIFT_DIR)/libsvm; cd -\

$@.c: libift libsvm $@.c
	make 
	gcc $(FLAGS) $@.c -o $(BIN)/$@  $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

clean:
	rm $(BIN)/*;

