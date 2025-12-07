#FLAGS= -g -O0 -Wall -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -O3 -Wall -fopenmp -pthread -std=gnu11 -pedantic
IFTLIB    = -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lm
IFTFLAGS  = -I$(NEWIFT_DIR)/include -I /System/Library/Frameworks/vecLib.framework/Headers/

libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\

$@.c: libift $@.c 
	make
	gcc $(FLAGS) $@.c -o $@ $(IFTFLAGS) $(IFTLIB); \

clean:
	rm $(BIN)/*;
