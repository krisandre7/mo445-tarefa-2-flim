# For compilation, you must install zlib2 and zlib1g-dev 

LIB=./lib
INCLUDE=./include
BIN=./bin
SRC=./src
OBJ=./obj
DEMO=./demo
#FLAGS= -g -O0 -D _DEBUG
FLAGS= -O3 -Wall -Wno-unused-result -std=gnu11

INCLUDES=-I$(INCLUDE) -I$(OPF_DIR)/include

all: liboldift 
	@echo
	@echo "liboldift.a built..."
	@echo "DONE."

liboldift: $(LIB)/liboldift.a



$(LIB)/liboldift.a: \
$(OBJ)/common.o \
$(OBJ)/comptime.o \
$(OBJ)/border.o \
$(OBJ)/heap.o \
$(OBJ)/gheap.o \
$(OBJ)/realheap.o \
$(OBJ)/queue.o \
$(OBJ)/sort.o \
$(OBJ)/set.o \
$(OBJ)/curve.o \
$(OBJ)/geometry.o \
$(OBJ)/algebra.o \
$(OBJ)/plane.o \
$(OBJ)/polynom.o \
$(OBJ)/image.o \
$(OBJ)/cimage.o \
$(OBJ)/dimage.o \
$(OBJ)/descriptor.o \
$(OBJ)/salience.o \
$(OBJ)/bas.o \
$(OBJ)/ss.o \
$(OBJ)/metrics.o \
$(OBJ)/scene.o \
$(OBJ)/spectrum.o \
$(OBJ)/adjacency.o \
$(OBJ)/adjacency3.o \
$(OBJ)/annimg.o \
$(OBJ)/annscn.o \
$(OBJ)/frame.o \
$(OBJ)/genift.o \
$(OBJ)/genift3.o \
$(OBJ)/matrix.o \
$(OBJ)/realmatrix.o \
$(OBJ)/mathematics.o \
$(OBJ)/mathematics3.o \
$(OBJ)/radiometric.o \
$(OBJ)/radiometric3.o \
$(OBJ)/filtering.o \
$(OBJ)/filtering3.o \
$(OBJ)/morphology.o \
$(OBJ)/morphology3.o \
$(OBJ)/segmentation.o \
$(OBJ)/segmentation3.o \
$(OBJ)/analysis.o \
$(OBJ)/analysis3.o \
$(OBJ)/context.o \
$(OBJ)/graphics.o \
$(OBJ)/shading.o \
$(OBJ)/shell.o \
$(OBJ)/normal.o \
$(OBJ)/raycasting.o \
$(OBJ)/splatting.o \
$(OBJ)/reslicing.o \
$(OBJ)/classify.o \
$(OBJ)/spline.o \
$(OBJ)/stackandfifo.o \
$(OBJ)/color.o \
$(OBJ)/cimg.o \
$(OBJ)/tensorscale.o \
$(OBJ)/complgraph.o \
$(OBJ)/prscene.o \
$(OBJ)/gqueue.o \
$(OBJ)/ctree.o \
$(OBJ)/mclouds.o \
$(OBJ)/interpolation.o \
$(OBJ)/interpolation3.o \
$(OBJ)/matching.o \
$(OBJ)/geometric3.o \
$(OBJ)/geometric2.o \
$(OBJ)/reslice3.o \
$(OBJ)/glist.o \
$(OBJ)/feature.o \
$(OBJ)/feature3.o \
$(OBJ)/msfiltering.o \
$(OBJ)/opfimg.o \
$(OBJ)/subgraph.o \
$(OBJ)/tira.o \
$(OBJ)/ctree3.o \
$(OBJ)/sgctree.o \
$(OBJ)/file_analyze.o \
$(OBJ)/bz2_lib.o \
$(OBJ)/msp.o \
$(OBJ)/compressed.o \
$(OBJ)/msps.o \
$(OBJ)/nifti1_io.o \
$(OBJ)/znzlib.o 
	ar csr $(LIB)/liboldift.a \
$(OBJ)/common.o \
$(OBJ)/comptime.o \
$(OBJ)/border.o \
$(OBJ)/heap.o \
$(OBJ)/gheap.o \
$(OBJ)/realheap.o \
$(OBJ)/queue.o \
$(OBJ)/sort.o \
$(OBJ)/set.o \
$(OBJ)/curve.o \
$(OBJ)/geometry.o \
$(OBJ)/algebra.o \
$(OBJ)/plane.o \
$(OBJ)/polynom.o \
$(OBJ)/image.o \
$(OBJ)/cimage.o \
$(OBJ)/dimage.o \
$(OBJ)/descriptor.o \
$(OBJ)/salience.o \
$(OBJ)/bas.o \
$(OBJ)/ss.o \
$(OBJ)/metrics.o \
$(OBJ)/scene.o \
$(OBJ)/spectrum.o \
$(OBJ)/adjacency.o \
$(OBJ)/adjacency3.o \
$(OBJ)/annimg.o \
$(OBJ)/annscn.o \
$(OBJ)/frame.o \
$(OBJ)/genift.o \
$(OBJ)/genift3.o \
$(OBJ)/matrix.o \
$(OBJ)/realmatrix.o \
$(OBJ)/mathematics.o \
$(OBJ)/mathematics3.o \
$(OBJ)/radiometric.o \
$(OBJ)/radiometric3.o \
$(OBJ)/filtering.o \
$(OBJ)/filtering3.o \
$(OBJ)/morphology.o \
$(OBJ)/morphology3.o \
$(OBJ)/segmentation.o \
$(OBJ)/segmentation3.o \
$(OBJ)/analysis.o \
$(OBJ)/analysis3.o \
$(OBJ)/context.o \
$(OBJ)/graphics.o \
$(OBJ)/shading.o \
$(OBJ)/shell.o \
$(OBJ)/normal.o \
$(OBJ)/raycasting.o \
$(OBJ)/splatting.o \
$(OBJ)/reslicing.o \
$(OBJ)/classify.o \
$(OBJ)/spline.o \
$(OBJ)/stackandfifo.o \
$(OBJ)/color.o \
$(OBJ)/cimg.o \
$(OBJ)/tensorscale.o \
$(OBJ)/complgraph.o \
$(OBJ)/prscene.o \
$(OBJ)/gqueue.o \
$(OBJ)/ctree.o \
$(OBJ)/mclouds.o \
$(OBJ)/interpolation.o \
$(OBJ)/interpolation3.o \
$(OBJ)/matching.o \
$(OBJ)/geometric3.o \
$(OBJ)/geometric2.o \
$(OBJ)/reslice3.o \
$(OBJ)/glist.o \
$(OBJ)/feature.o \
$(OBJ)/feature3.o \
$(OBJ)/msfiltering.o \
$(OBJ)/opfimg.o \
$(OBJ)/subgraph.o \
$(OBJ)/tira.o \
$(OBJ)/ctree3.o \
$(OBJ)/sgctree.o \
$(OBJ)/file_analyze.o \
$(OBJ)/bz2_lib.o \
$(OBJ)/msp.o \
$(OBJ)/compressed.o \
$(OBJ)/msps.o \
$(OBJ)/nifti1_io.o \
$(OBJ)/znzlib.o 

$(OBJ)/glist.o: $(SRC)/glist.c
	gcc $(FLAGS) -c $(SRC)/glist.c $(INCLUDES) -o $(OBJ)/glist.o

$(OBJ)/common.o: $(SRC)/common.c
	gcc $(FLAGS) -c $(SRC)/common.c $(INCLUDES) -o $(OBJ)/common.o

$(OBJ)/comptime.o: $(SRC)/comptime.c
	gcc $(FLAGS) -c $(SRC)/comptime.c $(INCLUDES) -o $(OBJ)/comptime.o

$(OBJ)/border.o: $(SRC)/border.c
	gcc $(FLAGS) -c $(SRC)/border.c $(INCLUDES) -o $(OBJ)/border.o

$(OBJ)/heap.o: $(SRC)/heap.c
	gcc $(FLAGS) -c $(SRC)/heap.c $(INCLUDES) -o $(OBJ)/heap.o

$(OBJ)/gheap.o: $(SRC)/gheap.c
	gcc $(FLAGS) -c $(SRC)/gheap.c $(INCLUDES) -o $(OBJ)/gheap.o

$(OBJ)/realheap.o: $(SRC)/realheap.c
	gcc $(FLAGS) -c $(SRC)/realheap.c $(INCLUDES) -o $(OBJ)/realheap.o

$(OBJ)/queue.o: $(SRC)/queue.c
	gcc $(FLAGS) -c $(SRC)/queue.c $(INCLUDES) -o $(OBJ)/queue.o

$(OBJ)/sort.o: $(SRC)/sort.c
	gcc $(FLAGS) -c $(SRC)/sort.c $(INCLUDES) -o $(OBJ)/sort.o

$(OBJ)/set.o: $(SRC)/set.c
	gcc $(FLAGS) -c $(SRC)/set.c $(INCLUDES) -o $(OBJ)/set.o

$(OBJ)/curve.o: $(SRC)/curve.c
	gcc $(FLAGS) -c $(SRC)/curve.c $(INCLUDES) -o $(OBJ)/curve.o

$(OBJ)/geometry.o: $(SRC)/geometry.c
	gcc $(FLAGS) -c $(SRC)/geometry.c $(INCLUDES) -o $(OBJ)/geometry.o

$(OBJ)/algebra.o: $(SRC)/algebra.c
	gcc $(FLAGS) -c $(SRC)/algebra.c $(INCLUDES) -o $(OBJ)/algebra.o

$(OBJ)/plane.o: $(SRC)/plane.c
	gcc $(FLAGS) -c $(SRC)/plane.c $(INCLUDES) -o $(OBJ)/plane.o

$(OBJ)/polynom.o: $(SRC)/polynom.c
	gcc $(FLAGS) -c $(SRC)/polynom.c $(INCLUDES) -o $(OBJ)/polynom.o

$(OBJ)/image.o: $(SRC)/image.c
	gcc $(FLAGS) -c $(SRC)/image.c $(INCLUDES) -o $(OBJ)/image.o

$(OBJ)/cimage.o: $(SRC)/cimage.c
	gcc $(FLAGS) -c $(SRC)/cimage.c $(INCLUDES) -o $(OBJ)/cimage.o

$(OBJ)/dimage.o: $(SRC)/dimage.c
	gcc $(FLAGS) -c $(SRC)/dimage.c $(INCLUDES) -o $(OBJ)/dimage.o

$(OBJ)/scene.o: $(SRC)/scene.c
	gcc $(FLAGS) -c $(SRC)/scene.c $(INCLUDES) -o $(OBJ)/scene.o

$(OBJ)/spectrum.o: $(SRC)/spectrum.c
	gcc $(FLAGS) -c $(SRC)/spectrum.c $(INCLUDES) -o $(OBJ)/spectrum.o

$(OBJ)/adjacency.o: $(SRC)/adjacency.c
	gcc $(FLAGS) -c $(SRC)/adjacency.c $(INCLUDES) -o $(OBJ)/adjacency.o

$(OBJ)/adjacency3.o: $(SRC)/adjacency3.c
	gcc $(FLAGS) -c $(SRC)/adjacency3.c $(INCLUDES) -o $(OBJ)/adjacency3.o

$(OBJ)/annimg.o: $(SRC)/annimg.c
	gcc $(FLAGS) -c $(SRC)/annimg.c $(INCLUDES) -o $(OBJ)/annimg.o

$(OBJ)/annscn.o: $(SRC)/annscn.c
	gcc $(FLAGS) -c $(SRC)/annscn.c $(INCLUDES) -o $(OBJ)/annscn.o

$(OBJ)/frame.o: $(SRC)/frame.c
	gcc $(FLAGS) -c $(SRC)/frame.c $(INCLUDES) -o $(OBJ)/frame.o

$(OBJ)/mathematics.o: $(SRC)/mathematics.c
	gcc $(FLAGS) -c $(SRC)/mathematics.c $(INCLUDES) -o $(OBJ)/mathematics.o

$(OBJ)/mathematics3.o: $(SRC)/mathematics3.c
	gcc $(FLAGS) -c $(SRC)/mathematics3.c $(INCLUDES) -o $(OBJ)/mathematics3.o

$(OBJ)/radiometric.o: $(SRC)/radiometric.c
	gcc $(FLAGS) -c $(SRC)/radiometric.c $(INCLUDES) -o $(OBJ)/radiometric.o

$(OBJ)/radiometric3.o: $(SRC)/radiometric3.c
	gcc $(FLAGS) -c $(SRC)/radiometric3.c $(INCLUDES) -o $(OBJ)/radiometric3.o

$(OBJ)/morphology.o: $(SRC)/morphology.c
	gcc $(FLAGS) -c $(SRC)/morphology.c $(INCLUDES) -o $(OBJ)/morphology.o

$(OBJ)/morphology3.o: $(SRC)/morphology3.c
	gcc $(FLAGS) -c $(SRC)/morphology3.c $(INCLUDES) -o $(OBJ)/morphology3.o

$(OBJ)/segmentation.o: $(SRC)/segmentation.c
	gcc $(FLAGS) -c $(SRC)/segmentation.c $(INCLUDES) -o $(OBJ)/segmentation.o

$(OBJ)/segmentation3.o: $(SRC)/segmentation3.c
	gcc $(FLAGS) -c $(SRC)/segmentation3.c $(INCLUDES) -o $(OBJ)/segmentation3.o

$(OBJ)/analysis.o: $(SRC)/analysis.c
	gcc $(FLAGS) -c $(SRC)/analysis.c $(INCLUDES) -o $(OBJ)/analysis.o

$(OBJ)/analysis3.o: $(SRC)/analysis3.c
	gcc $(FLAGS) -c $(SRC)/analysis3.c $(INCLUDES) -o $(OBJ)/analysis3.o

$(OBJ)/filtering.o: $(SRC)/filtering.c
	gcc $(FLAGS) -c $(SRC)/filtering.c $(INCLUDES) -o $(OBJ)/filtering.o

$(OBJ)/filtering3.o: $(SRC)/filtering3.c
	gcc $(FLAGS) -c $(SRC)/filtering3.c $(INCLUDES) -o $(OBJ)/filtering3.o

$(OBJ)/context.o: $(SRC)/context.c
	gcc $(FLAGS) -c $(SRC)/context.c $(INCLUDES) -o $(OBJ)/context.o

$(OBJ)/graphics.o: $(SRC)/graphics.c
	gcc $(FLAGS) -c $(SRC)/graphics.c $(INCLUDES) -o $(OBJ)/graphics.o

$(OBJ)/shading.o: $(SRC)/shading.c
	gcc $(FLAGS) -c $(SRC)/shading.c $(INCLUDES) -o $(OBJ)/shading.o

$(OBJ)/shell.o: $(SRC)/shell.c
	gcc $(FLAGS) -c $(SRC)/shell.c $(INCLUDES) -o $(OBJ)/shell.o

$(OBJ)/normal.o: $(SRC)/normal.c
	gcc $(FLAGS) -c $(SRC)/normal.c $(INCLUDES) -o $(OBJ)/normal.o

$(OBJ)/raycasting.o: $(SRC)/raycasting.c
	gcc $(FLAGS) -c $(SRC)/raycasting.c $(INCLUDES) -o $(OBJ)/raycasting.o

$(OBJ)/splatting.o: $(SRC)/splatting.c
	gcc $(FLAGS) -c $(SRC)/splatting.c $(INCLUDES) -o $(OBJ)/splatting.o

$(OBJ)/reslicing.o: $(SRC)/reslicing.c
	gcc $(FLAGS) -c $(SRC)/reslicing.c $(INCLUDES) -o $(OBJ)/reslicing.o

$(OBJ)/genift.o: $(SRC)/genift.c
	gcc $(FLAGS) -c $(SRC)/genift.c $(INCLUDES) -o $(OBJ)/genift.o

$(OBJ)/genift3.o: $(SRC)/genift3.c
	gcc $(FLAGS) -c $(SRC)/genift3.c $(INCLUDES) -o $(OBJ)/genift3.o

$(OBJ)/descriptor.o: $(SRC)/descriptor.c
	gcc $(FLAGS) -c $(SRC)/descriptor.c $(INCLUDES) -o $(OBJ)/descriptor.o

$(OBJ)/salience.o: $(SRC)/salience.c
	gcc $(FLAGS) -c $(SRC)/salience.c $(INCLUDES) -o $(OBJ)/salience.o

$(OBJ)/bas.o: $(SRC)/bas.c
	gcc $(FLAGS) -c $(SRC)/bas.c $(INCLUDES) -o $(OBJ)/bas.o

$(OBJ)/ss.o: $(SRC)/ss.c
	gcc $(FLAGS) -c $(SRC)/ss.c $(INCLUDES) -o $(OBJ)/ss.o

$(OBJ)/metrics.o: $(SRC)/metrics.c
	gcc $(FLAGS) -c $(SRC)/metrics.c $(INCLUDES) -o $(OBJ)/metrics.o

$(OBJ)/matrix.o: $(SRC)/matrix.c
	gcc $(FLAGS) -c $(SRC)/matrix.c $(INCLUDES) -o $(OBJ)/matrix.o

$(OBJ)/realmatrix.o: $(SRC)/realmatrix.c
	gcc $(FLAGS) -c $(SRC)/realmatrix.c $(INCLUDES) -o $(OBJ)/realmatrix.o

$(OBJ)/classify.o: $(SRC)/classify.c
	gcc $(FLAGS) -c $(SRC)/classify.c $(INCLUDES) -o $(OBJ)/classify.o

$(OBJ)/spline.o: $(SRC)/spline.c
	gcc $(FLAGS) -c $(SRC)/spline.c $(INCLUDES) -o $(OBJ)/spline.o

$(OBJ)/stackandfifo.o: $(SRC)/stackandfifo.c
	gcc $(FLAGS) -c $(SRC)/stackandfifo.c $(INCLUDES) -o $(OBJ)/stackandfifo.o

$(OBJ)/cimg.o: $(SRC)/cimg.c
	gcc $(FLAGS) -c $(SRC)/cimg.c $(INCLUDES) -o $(OBJ)/cimg.o

$(OBJ)/color.o: $(SRC)/color.c
	gcc $(FLAGS) -c $(SRC)/color.c $(INCLUDES) -o $(OBJ)/color.o

$(OBJ)/tensorscale.o: $(SRC)/tensorscale.c
	gcc $(FLAGS) -c $(SRC)/tensorscale.c $(INCLUDES) -o $(OBJ)/tensorscale.o

$(OBJ)/complgraph.o: $(SRC)/complgraph.c
	gcc $(FLAGS) -c $(SRC)/complgraph.c $(INCLUDES) -o $(OBJ)/complgraph.o

$(OBJ)/prscene.o: $(SRC)/prscene.c
	gcc $(FLAGS) -c $(SRC)/prscene.c $(INCLUDES) -o $(OBJ)/prscene.o

$(OBJ)/gqueue.o: $(SRC)/gqueue.c
	gcc $(FLAGS) -c $(SRC)/gqueue.c $(INCLUDES) -o $(OBJ)/gqueue.o

$(OBJ)/ctree.o: $(SRC)/ctree.c
	gcc $(FLAGS) -c $(SRC)/ctree.c $(INCLUDES) -o $(OBJ)/ctree.o

$(OBJ)/mclouds.o: $(SRC)/mclouds.c
	gcc $(FLAGS) -c $(SRC)/mclouds.c $(INCLUDES) -o $(OBJ)/mclouds.o

$(OBJ)/interpolation.o: $(SRC)/interpolation.c
	gcc $(FLAGS) -c $(SRC)/interpolation.c $(INCLUDES) -o $(OBJ)/interpolation.o

$(OBJ)/interpolation3.o: $(SRC)/interpolation3.c
	gcc $(FLAGS) -c $(SRC)/interpolation3.c $(INCLUDES) -o $(OBJ)/interpolation3.o

$(OBJ)/matching.o: $(SRC)/matching.c
	gcc $(FLAGS) -c $(SRC)/matching.c $(INCLUDES) -o $(OBJ)/matching.o

$(OBJ)/geometric3.o: $(SRC)/geometric3.c
	gcc $(FLAGS) -c $(SRC)/geometric3.c $(INCLUDES) -o $(OBJ)/geometric3.o

$(OBJ)/geometric2.o: $(SRC)/geometric2.c
	gcc $(FLAGS) -c $(SRC)/geometric2.c $(INCLUDES) -o $(OBJ)/geometric2.o

$(OBJ)/registration.o: $(SRC)/registration.c
	gcc $(FLAGS) -c $(SRC)/registration.c $(INCLUDES) -o $(OBJ)/registration.o

$(OBJ)/reslice3.o: $(SRC)/reslice3.c
	gcc $(FLAGS) -c $(SRC)/reslice3.c $(INCLUDES) -o $(OBJ)/reslice3.o

$(OBJ)/feature.o: $(SRC)/feature.c
	gcc $(FLAGS) -c $(SRC)/feature.c $(INCLUDES) -o $(OBJ)/feature.o

$(OBJ)/feature3.o: $(SRC)/feature3.c
	gcc $(FLAGS) -c $(SRC)/feature3.c $(INCLUDES) -o $(OBJ)/feature3.o

$(OBJ)/msfiltering.o: $(SRC)/msfiltering.c
	gcc $(FLAGS) -c $(SRC)/msfiltering.c $(INCLUDES) -o $(OBJ)/msfiltering.o

$(OBJ)/opfimg.o: $(SRC)/opfimg.c
	gcc $(FLAGS) -c $(SRC)/opfimg.c $(INCLUDES) -o $(OBJ)/opfimg.o

$(OBJ)/subgraph.o: $(SRC)/subgraph.c
	gcc $(FLAGS) -c $(SRC)/subgraph.c $(INCLUDES) -o $(OBJ)/subgraph.o
$(OBJ)/tira.o: $(SRC)/tira.c
	gcc $(FLAGS) -c $(SRC)/tira.c $(INCLUDES) -o $(OBJ)/tira.o

$(OBJ)/ctree3.o: $(SRC)/ctree3.c
	gcc $(FLAGS) -c $(SRC)/ctree3.c $(INCLUDES) -o $(OBJ)/ctree3.o

$(OBJ)/sgctree.o: $(SRC)/sgctree.c
	gcc $(FLAGS) -c $(SRC)/sgctree.c $(INCLUDES) -o $(OBJ)/sgctree.o

$(OBJ)/file_analyze.o: $(SRC)/file_analyze.c
	gcc $(FLAGS) -c $(SRC)/file_analyze.c $(INCLUDES) -o $(OBJ)/file_analyze.o

$(OBJ)/bz2_lib.o: $(SRC)/bz2_lib.c
	gcc $(FLAGS) -c $(SRC)/bz2_lib.c $(INCLUDES) -o $(OBJ)/bz2_lib.o

$(OBJ)/msp.o: $(SRC)/msp.c
	gcc $(FLAGS) -c $(SRC)/msp.c $(INCLUDES) -o $(OBJ)/msp.o

$(OBJ)/compressed.o: $(SRC)/compressed.c
	gcc $(FLAGS) -c $(SRC)/compressed.c $(INCLUDES) -o $(OBJ)/compressed.o

$(OBJ)/msps.o: $(SRC)/msps.c
	gcc $(FLAGS) -c $(SRC)/msps.c $(INCLUDES) -o $(OBJ)/msps.o

$(OBJ)/znzlib.o: $(SRC)/znzlib.c
	gcc $(FLAGS) -c $(SRC)/znzlib.c $(INCLUDES) -o $(OBJ)/znzlib.o 

$(OBJ)/nifti1_io.o: $(SRC)/nifti1_io.c
	gcc $(FLAGS) -c $(SRC)/nifti1_io.c $(INCLUDES) -o $(OBJ)/nifti1_io.o

clean:
	rm -f $(LIB)/lib*.a; rm -f $(OBJ)/*.o; rm -f $(BIN)/*




