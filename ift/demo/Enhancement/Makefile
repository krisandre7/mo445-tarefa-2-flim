
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

enhancement: libift libsvm enhancement.c iftTrainEdges.o common.o
	make 
	gcc $(FLAGS) enhancement.c iftTrainEdges.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

enhancementwg: libift libsvm enhancementwg.c iftTrainEdges.o common.o
	make 
	gcc $(FLAGS) enhancementwg.c iftTrainEdges.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

borderedges: libift borderedges.c iftTrainEdges.o common.o
	make 
	gcc $(FLAGS) borderedges.c iftTrainEdges.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

borderpixels: libift borderpixels.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) borderpixels.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

datasetpixels: libift libsvm datasetpixels.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) datasetpixels.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

evaluatepixels: libift libsvm evaluatepixels.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) evaluatepixels.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

borderclassify: libift libsvm borderclassify.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) borderclassify.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

labelpixels: libift libsvm labelpixels.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) labelpixels.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

wiimage: libift libsvm wiimage.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) wiimage.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \


gimage: libift libsvm gimage.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) gimage.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

getgrab: libift libsvm getgrab.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) getgrab.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

iftTestGTWG: libift libsvm iftTestGTWG.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) iftTestGTWG.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

iftSupBorderClassify: libift libsvm iftSupBorderClassify.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) iftSupBorderClassify.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \


iftGetBasins: libift libsvm iftGetBasins.c iftTrainPixels.o common.o
	make 
	gcc $(FLAGS) iftGetBasins.c iftTrainPixels.o common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

msconvnet: libift libsvm msconvnet.c common.o
	make 
	gcc $(FLAGS) msconvnet.c common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

borderstatistics: libift borderstatistics.c common.o
	make 
	gcc $(FLAGS) borderstatistics.c common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

standarddimensions: libift standarddimensions.c common.o
	make 
	gcc $(FLAGS) standarddimensions.c common.o -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB) $(LIBSVM) ; \

common.o: common.c
	gcc $(FLAGS) -c common.c $(IFTFLAGS) -o common.o 

iftTrainEdges.o: iftTrainEdges.c
	gcc $(FLAGS) -c iftTrainEdges.c  $(IFTFLAGS) -o iftTrainEdges.o 

iftTrainPixels.o: iftTrainPixels.c
	gcc $(FLAGS) -c iftTrainPixels.c $(IFTFLAGS) -o iftTrainPixels.o 

$@.c: libift $@.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB); \

all: 
	make enhancement 
	make enhancementwg 
	make borderedges 
	make borderpixels 
	make datasetpixels
	make evaluatepixels
	make borderclassify
	make msconvnet
	make standarddimensions

clean:
	rm $(BIN)/*;
	rm ./*.o
