C_FLAGS=-std=c99
LINK_OPTS=-L$(LIBSVM) -lsvm -lstdc++ -llapack -lcblas -lblas -fopenmp
LIBDGC_DIR=$(NEWIFT_DIR)/libdgc/

libift:
	@echo Building libift
	cd $(NEWIFT_DIR); make -C $(NEWIFT_DIR); cd -\

libdgc:
	@echo Building libdgc
	cd $(LIBDGC_DIR); make -C $(LIBDGC_DIR); cd -\

main: libift
	gcc -c MainRelax.c -o obj/mainrelax.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	gcc -c common.c -o obj/common.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	gcc -o IsMoya obj/mainrelax.o obj/common.o -L $(NEWIFT_DIR)/lib -l ift $(LINK_OPTS)

maindgcv: libift libdgc
	gcc -c common.c -o obj/common.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	g++ -c mainDGCV.cpp -o obj/maindgcv.o -I $(NEWIFT_DIR)/include -I$(LIBDGC_DIR) -Wno-write-strings
	g++ -o maindgcv obj/common.o obj/maindgcv.o -L $(NEWIFT_DIR)/lib -l ift $(LINK_OPTS)

maindgcsv: libift libdgc
	gcc -c common.c -o obj/common.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	g++ -c mainDGCSV.cpp -o obj/maindgcsv.o -I $(NEWIFT_DIR)/include -I$(LIBDGC_DIR) -Wno-write-strings
	g++ -o maindgcsv obj/common.o obj/maindgcsv.o -L $(NEWIFT_DIR)/lib -l ift $(LINK_OPTS)

clean:
	rm -rf ./obj/*.o