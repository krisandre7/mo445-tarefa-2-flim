BIN=../../bin
OBJ=obj

#FLAGS= -ggdb -O0 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
#FLAGS= -g -O0 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
FLAGS= -O3 -Wall -Wno-unused-result -fopenmp -pthread -std=gnu11 -pedantic
IFTLIB    = -L$(NEWIFT_DIR)/lib -lift -llapack -lblas  -lm -L/lib64/atlas-sse3/ -L/usr/lib/atlas-base -lcblas -lpng -lz
IFTFLAGS  = -I$(NEWIFT_DIR)/include 
LIBSVM    = -L$(NEWIFT_DIR)/libsvm -lsvm -lstdc++ 

libift:
	cd ../../; $(MAKE) -C $(NEWIFT_DIR); cd -\

libsvm:
	cd ../../libsvm; $(MAKE) -C $(NEWIFT_DIR)/libsvm; cd -\

$@.c: libift $@.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(IFTLIB); \

hutils: 
	mkdir -p $(OBJ)
	$(CC) $(FLAGS) -c hutils.c -o $(OBJ)/hutils.o $(IFTFLAGS) ; \

iftHierarchicalSegmentation: 
	mkdir -p $(OBJ)
	$(CC) $(FLAGS) -c iftHierarchicalSegmentation.c -o $(OBJ)/iftHierarchicalSegmentation.o $(IFTFLAGS) ; \

iftContextualClassification: 
	mkdir -p $(OBJ)
	$(CC) $(FLAGS) -c iftContextualClassification.c -o $(OBJ)/iftContextualClassification.o $(IFTFLAGS) ; \

iftHierarchicalClustering: 
	mkdir -p $(OBJ)
	$(CC) $(FLAGS) -c iftHierarchicalClustering.c -o $(OBJ)/iftHierarchicalClustering.o $(IFTFLAGS) ; \

iftHierarchyCut: libift libsvm hutils iftHierarchicalSegmentation iftHierarchyCut.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftRunContextualClassification: libift hutils libsvm iftContextualClassification iftRunContextualClassification.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftBaselineActiveLearningMCLU: libift hutils libsvm iftBaselineActiveLearningMCLU.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftBaselineClusteringActiveLearningMCLU: libift hutils libsvm iftBaselineClusteringActiveLearningMCLU.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftRegionActiveLearningMCLU: libift hutils libsvm iftRegionActiveLearningMCLU.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftHierarchyCutTopDown: libift hutils iftHierarchicalClustering iftHierarchyCutTopDown.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)

iftComputeHIndexForAllClasses: libift hutils iftHierarchicalClustering iftComputeHIndexForAllClasses.c 
	make
	gcc $(FLAGS) $@.c -g -O0 -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)

iftComputeHIndexSupervisedCut: libift hutils iftHierarchicalClustering iftComputeHIndexSupervisedCut.c 
	make
	gcc $(FLAGS) $@.c -g -O0 -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)

iftClusteringAccuracy: libift hutils iftHierarchicalClustering iftClusteringAccuracy.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)

iftActiveClusteringCut: libift hutils iftHierarchicalClustering iftActiveClusteringCut.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)

iftClusteringAccuracyByRegion: libift hutils  iftClusteringAccuracyByRegion.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)

iftComputeClassificationMap: libift hutils libsvm iftComputeClassificationMap.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftComputeClassificationMapPixel: libift hutils libsvm iftComputeClassificationMapPixel.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftEvaluateBoundaryClassifier: libift hutils libsvm iftHierarchicalSegmentation iftEvaluateBoundaryClassifier.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftHierarchyCutSupervised: libift hutils libsvm iftHierarchicalSegmentation iftHierarchyCutSupervised.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftComputeClassificationMapFromRegions: libift hutils libsvm iftComputeClassificationMapFromRegions.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

iftComputeClassificationMapUsingInputDataSet: libift hutils libsvm iftComputeClassificationMapUsingInputDataSet.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)


iftComputeClassificationMapFromRegionsUsingInputDataSet: libift hutils libsvm iftComputeClassificationMapFromRegionsUsingInputDataSet.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)	


iftRandomAnnotationClusteringPropagation: libift hutils iftHierarchicalClustering iftRandomAnnotationClusteringPropagation.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB); \
	rm -rf $(OBJ)


iftComputeClassificationMapPropagatingLabels: libift hutils libsvm iftComputeClassificationMapPropagatingLabels.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)


iftComputeClassificationMapWritingWeights: libift hutils libsvm iftComputeClassificationMapWritingWeights.c 
	make
	gcc $(FLAGS) $@.c -o $(BIN)/$@ $(IFTFLAGS) $(OBJ)/*.o $(IFTLIB) $(LIBSVM); \
	rm -rf $(OBJ)

clean:
	rm $(BIN)/*;
