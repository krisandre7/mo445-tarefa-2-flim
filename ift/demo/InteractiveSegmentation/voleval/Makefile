C_FLAGS=-std=c99
LINK_OPTS=-L$(LIBSVM) -lsvm -lstdc++ -llapack -lcblas -lblas -fopenmp -latlas
LIBDGC_DIR=$(NEWIFT_DIR)/libdgc/

libift:
	@echo Building libift
	cd $(NEWIFT_DIR); make -C $(NEWIFT_DIR); cd -\

libdgc:
	@echo Building libdgc
	cd $(LIBDGC_DIR); make -C $(LIBDGC_DIR); cd -\

main: libift
	gcc -c main.c -o obj/main.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	gcc -c common.c -o obj/common.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	gcc -o main obj/main.o obj/common.o -L $(NEWIFT_DIR)/lib -l ift $(LINK_OPTS)

mainphotos: libift
	gcc -c main.c -o obj/mainphotos.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	gcc -c common.c -o obj/commonphotos.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	gcc -o mainphotos obj/mainphotos.o obj/commonphotos.o -L $(NEWIFT_DIR)/lib -l ift $(LINK_OPTS)

maindgc: libift libdgc
	gcc -c common.c -o obj/common.o -I $(NEWIFT_DIR)/include $(C_FLAGS)
	g++ -c mainDGC.cpp -o obj/maindgc.o -I $(NEWIFT_DIR)/include -I$(LIBDGC_DIR) -Wno-write-strings
	g++ -o maindgc obj/common.o obj/maindgc.o -L $(NEWIFT_DIR)/lib -l ift $(LINK_OPTS)
	