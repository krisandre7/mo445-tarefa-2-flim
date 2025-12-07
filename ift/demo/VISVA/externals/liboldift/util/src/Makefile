BIN=../../bin

#FLAGS= -g -Wall 
FLAGS= -O3 -Wall -Wno-unused-result
IFTLIB    = -L$(OPF_DIR)/lib -lOPF -L$(IFT_DIR)/lib -lift -lm -lz
IFTFLAGS  = -I$(IFT_DIR)/include -I$(OPF_DIR)/include


libift:
	$(MAKE) -C $(IFT_DIR); \

libopf:
	$(MAKE) -C $(OPF_DIR) opf-ift ; \

$@.c: libift libopf $@.c 
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) ; \

clean:
	rm $(BIN)/*; rm *.pgm; rm *~; rm *.ppm










