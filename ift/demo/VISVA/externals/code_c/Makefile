all: libcode_c # scn2png sceneop flip_scn standalone_rigid_registration rotate_scn rotate_scn2 crop_scn

progs: scn2png sceneop flip_scn standalone_rigid_registration rotate_scn rotate_scn2


LIB= ./lib
HDR=./include
SRC=./src
OBJ=./obj

#Rules
libcode_c: $(LIB)/libcode_c.a
	echo "libcode_c.a built..."

$(LIB)/libcode_c.a: \
$(OBJ)/fio.o \
$(OBJ)/adjacency_addons.o \
$(OBJ)/scene_addons.o \
$(OBJ)/string_addons.o \
$(OBJ)/evaluation3.o \
$(OBJ)/floatadjacency3.o \
$(OBJ)/fuzzycloud3.o \
$(OBJ)/labeledfeatures.o \
$(OBJ)/mmorphology.o \
$(OBJ)/mri.o \
$(OBJ)/objgroup.o \
$(OBJ)/pyramid3.o \
$(OBJ)/radiometric_addons.o \
$(OBJ)/rce.o \
$(OBJ)/realarray.o \
$(OBJ)/seedmap3.o \
$(OBJ)/gradient.o \
$(OBJ)/methods.o \
$(OBJ)/preproc.o \
$(OBJ)/posproc.o \
$(OBJ)/shared.o \
$(OBJ)/genset.o \
$(OBJ)/view.o \
$(OBJ)/featmap.o \
$(OBJ)/hashtable.o \
$(OBJ)/xmldoc.o \
$(OBJ)/filelist.o \
$(OBJ)/arraylist.o \
$(OBJ)/cloud3.o \
$(OBJ)/markerlist.o \
$(OBJ)/segmobject.o \
$(OBJ)/simetria.o \
$(OBJ)/priorityqueue.o \
$(OBJ)/rigidregistration3.o \
$(OBJ)/inhomogeneity.o \
$(OBJ)/file_bia.o \
$(OBJ)/braincluster_opf.o \
$(OBJ)/braincluster_spectral.o \
$(OBJ)/braincluster_atlas.o \
$(OBJ)/orderlist.o \
$(OBJ)/color_reg.o \
$(OBJ)/selection3.o \
$(OBJ)/measures.o \
$(OBJ)/msp.o \
$(OBJ)/myopf3.o \
$(OBJ)/processors.o
	ar csr $(LIB)/libcode_c.a \
$(OBJ)/fio.o \
$(OBJ)/adjacency_addons.o \
$(OBJ)/scene_addons.o \
$(OBJ)/string_addons.o \
$(OBJ)/evaluation3.o \
$(OBJ)/floatadjacency3.o \
$(OBJ)/fuzzycloud3.o \
$(OBJ)/labeledfeatures.o \
$(OBJ)/mmorphology.o \
$(OBJ)/mri.o \
$(OBJ)/objgroup.o \
$(OBJ)/pyramid3.o \
$(OBJ)/radiometric_addons.o \
$(OBJ)/rce.o \
$(OBJ)/realarray.o \
$(OBJ)/seedmap3.o \
$(OBJ)/gradient.o \
$(OBJ)/methods.o \
$(OBJ)/preproc.o \
$(OBJ)/posproc.o \
$(OBJ)/shared.o \
$(OBJ)/genset.o \
$(OBJ)/view.o \
$(OBJ)/featmap.o \
$(OBJ)/hashtable.o \
$(OBJ)/xmldoc.o \
$(OBJ)/filelist.o \
$(OBJ)/arraylist.o \
$(OBJ)/cloud3.o \
$(OBJ)/markerlist.o \
$(OBJ)/segmobject.o \
$(OBJ)/simetria.o \
$(OBJ)/priorityqueue.o \
$(OBJ)/rigidregistration3.o \
$(OBJ)/inhomogeneity.o \
$(OBJ)/file_bia.o \
$(OBJ)/braincluster_opf.o \
$(OBJ)/braincluster_spectral.o \
$(OBJ)/braincluster_atlas.o \
$(OBJ)/orderlist.o \
$(OBJ)/color_reg.o \
$(OBJ)/selection3.o \
$(OBJ)/measures.o \
$(OBJ)/msp.o \
$(OBJ)/myopf3.o \
$(OBJ)/processors.o


$(OBJ)/fio.o: $(SRC)/fio.c $(HDR)/fio.h
	$(CC) $(CC_FLAGS) -c $(SRC)/fio.c -I$(HDR) \
	-o $(OBJ)/fio.o

$(OBJ)/scene_addons.o: $(SRC)/scene_addons.c $(HDR)/scene_addons.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/scene_addons.c -I$(HDR) \
	-o $(OBJ)/scene_addons.o

$(OBJ)/selection3.o: $(SRC)/selection3.c $(HDR)/selection3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/selection3.c -I$(HDR) \
	-o $(OBJ)/selection3.o

$(OBJ)/rce.o: $(SRC)/rce.c $(HDR)/rce.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/rce.c -I$(HDR) \
	-o $(OBJ)/rce.o

$(OBJ)/labeledfeatures.o: $(SRC)/labeledfeatures.c $(HDR)/labeledfeatures.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/labeledfeatures.c -I$(HDR) \
	-o $(OBJ)/labeledfeatures.o

$(OBJ)/seedmap3.o: $(SRC)/seedmap3.c $(HDR)/seedmap3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/seedmap3.c -I$(HDR) \
	-o $(OBJ)/seedmap3.o

$(OBJ)/fuzzycloud3.o: $(SRC)/fuzzycloud3.c $(HDR)/fuzzycloud3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/fuzzycloud3.c -I$(HDR) \
	-o $(OBJ)/fuzzycloud3.o

$(OBJ)/string_addons.o: $(SRC)/string_addons.c $(HDR)/string_addons.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/string_addons.c -I$(HDR) \
	-o $(OBJ)/string_addons.o

$(OBJ)/adjacency_addons.o: $(SRC)/adjacency_addons.c $(HDR)/adjacency_addons.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/adjacency_addons.c -I$(HDR) \
	-o $(OBJ)/adjacency_addons.o

$(OBJ)/mri.o: $(SRC)/mri.c $(HDR)/mri.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/mri.c -I$(HDR) \
	-o $(OBJ)/mri.o

$(OBJ)/radiometric_addons.o: $(SRC)/radiometric_addons.c $(HDR)/radiometric_addons.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/radiometric_addons.c -I$(HDR) \
	-o $(OBJ)/radiometric_addons.o

$(OBJ)/pyramid3.o: $(SRC)/pyramid3.c $(HDR)/pyramid3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/pyramid3.c -I$(HDR) \
	-o $(OBJ)/pyramid3.o

$(OBJ)/realarray.o: $(SRC)/realarray.c $(HDR)/realarray.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/realarray.c -I$(HDR) \
	-o $(OBJ)/realarray.o

$(OBJ)/evaluation3.o: $(SRC)/evaluation3.c $(HDR)/evaluation3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/evaluation3.c -I$(HDR) \
	-o $(OBJ)/evaluation3.o

$(OBJ)/objgroup.o: $(SRC)/objgroup.c $(HDR)/objgroup.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/objgroup.c -I$(HDR) \
	-o $(OBJ)/objgroup.o

$(OBJ)/floatadjacency3.o: $(SRC)/floatadjacency3.c $(HDR)/floatadjacency3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/floatadjacency3.c -I$(HDR) \
	-o $(OBJ)/floatadjacency3.o

$(OBJ)/mmorphology.o: $(SRC)/mmorphology.c $(HDR)/mmorphology.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/mmorphology.c -I$(HDR) \
	-o $(OBJ)/mmorphology.o

$(OBJ)/gradient.o: $(SRC)/gradient.c $(HDR)/gradient.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/gradient.c -I$(HDR) \
	-o $(OBJ)/gradient.o

$(OBJ)/methods.o: $(SRC)/methods.c $(HDR)/methods.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/methods.c -I$(HDR) \
	-o $(OBJ)/methods.o

$(OBJ)/preproc.o: $(SRC)/preproc.c $(HDR)/preproc.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/preproc.c -I$(HDR) \
	-o $(OBJ)/preproc.o

$(OBJ)/posproc.o: $(SRC)/posproc.c $(HDR)/posproc.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/posproc.c -I$(HDR) \
	-o $(OBJ)/posproc.o

$(OBJ)/shared.o: $(SRC)/shared.c $(HDR)/shared.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/shared.c -I$(HDR) \
	-o $(OBJ)/shared.o

$(OBJ)/genset.o: $(SRC)/genset.c $(HDR)/genset.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/genset.c -I$(HDR) \
	-o $(OBJ)/genset.o

$(OBJ)/view.o: $(SRC)/view.c $(HDR)/view.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/view.c -I$(HDR) \
	-o $(OBJ)/view.o

$(OBJ)/featmap.o: $(SRC)/featmap.c $(HDR)/featmap.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/featmap.c -I$(HDR) \
	-o $(OBJ)/featmap.o

$(OBJ)/hashtable.o: $(SRC)/hashtable.c $(HDR)/hashtable.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/hashtable.c -I$(HDR) \
	-o $(OBJ)/hashtable.o

$(OBJ)/xmldoc.o: $(SRC)/xmldoc.c $(HDR)/xmldoc.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/xmldoc.c -I$(HDR) \
	-o $(OBJ)/xmldoc.o

$(OBJ)/filelist.o: $(SRC)/filelist.c $(HDR)/filelist.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/filelist.c -I$(HDR) \
	-o $(OBJ)/filelist.o

$(OBJ)/arraylist.o: $(SRC)/arraylist.c $(HDR)/arraylist.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/arraylist.c -I$(HDR) \
	-o $(OBJ)/arraylist.o

$(OBJ)/cloud3.o: $(SRC)/cloud3.c $(HDR)/cloud3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/cloud3.c -I$(HDR) \
	-o $(OBJ)/cloud3.o

$(OBJ)/markerlist.o: $(SRC)/markerlist.c $(HDR)/markerlist.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/markerlist.c -I$(HDR) \
	-o $(OBJ)/markerlist.o

$(OBJ)/segmobject.o: $(SRC)/segmobject.c $(HDR)/segmobject.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/segmobject.c -I$(HDR) \
	-o $(OBJ)/segmobject.o

$(OBJ)/simetria.o: $(SRC)/simetria.c $(HDR)/simetria.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/simetria.c \
		-I$(HDR) -o $(OBJ)/simetria.o

$(OBJ)/priorityqueue.o: $(SRC)/priorityqueue.c $(HDR)/priorityqueue.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/priorityqueue.c -I$(HDR) \
	-o $(OBJ)/priorityqueue.o

$(OBJ)/rigidregistration3.o: $(SRC)/rigidregistration3.c $(HDR)/rigidregistration3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/rigidregistration3.c -I$(HDR) \
	-o $(OBJ)/rigidregistration3.o

$(OBJ)/inhomogeneity.o: $(SRC)/inhomogeneity.c $(HDR)/inhomogeneity.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(CODE_C_INCLUDE) -c -lpthread $(SRC)/inhomogeneity.c -I$(HDR) \
	-o $(OBJ)/inhomogeneity.o

$(OBJ)/file_bia.o: $(SRC)/file_bia.c $(HDR)/file_bia.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/file_bia.c -I$(HDR) \
	-o $(OBJ)/file_bia.o

$(OBJ)/braincluster_opf.o: $(SRC)/braincluster_opf.c
	$(CC) $(CC_FLAGS) $(CODE_C_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c -lpthread $(SRC)/braincluster_opf.c -I$(HDR) \
	-o $(OBJ)/braincluster_opf.o

$(OBJ)/braincluster_spectral.o: $(SRC)/braincluster_spectral.c
	$(CC) $(CC_FLAGS) $(CODE_C_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c -lpthread $(SRC)/braincluster_spectral.c -I$(HDR) \
	-o $(OBJ)/braincluster_spectral.o

$(OBJ)/braincluster_atlas.o: $(SRC)/braincluster_atlas.c
	$(CC) $(CC_FLAGS) $(CODE_C_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c -lpthread $(SRC)/braincluster_atlas.c -I$(HDR) \
	-o $(OBJ)/braincluster_atlas.o

$(OBJ)/orderlist.o: $(SRC)/orderlist.c $(HDR)/orderlist.h
	$(CC) $(CC_FLAGS) $(CODE_C_INCLUDE) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/orderlist.c -I$(HDR) \
	-o $(OBJ)/orderlist.o

$(OBJ)/color_reg.o: $(SRC)/color_reg.c $(HDR)/color_reg.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/color_reg.c -I$(HDR) \
	-o $(OBJ)/color_reg.o

$(OBJ)/myopf3.o: $(SRC)/myopf3.c $(HDR)/myopf3.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/myopf3.c -I$(HDR) \
	-o $(OBJ)/myopf3.o

$(OBJ)/measures.o: $(SRC)/measures.c $(HDR)/measures.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/measures.c -I$(HDR) \
	-o $(OBJ)/measures.o

$(OBJ)/msp.o: $(SRC)/msp.c $(HDR)/msp.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/msp.c -I$(HDR) \
	-o $(OBJ)/msp.o

$(OBJ)/processors.o: $(SRC)/processors.c $(HDR)/processors.h
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/processors.c -I$(HDR) \
	-o $(OBJ)/processors.o

standalone_rigid_registration: 
	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
	-D_RIGID_REGISTRATION_STANDALONE_ \
	$(SRC)/rigidregistration3.c -I$(HDR) \
	$(OPF_LIB) $(OLDIFT_LIB) -lm \
	-o ../../bin/rigid_registration_bin

# scn2png:
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(SRC)/scn2png.c \
# 	$(OPF_LIB) $(OLDIFT_LIB) -I$(HDR) -lm \
# 	-o ../../bin/scn2png

# sceneop: libcode_c
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -I$(HDR) $(SRC)/sceneop.c \
# 	-L. -lcode_c $(OPF_LIB) $(OLDIFT_LIB) -lm -pthread \
# 	-o ../../bin/sceneop

# crop_scn:
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(SRC)/crop_scn.c \
# 	$(OPF_LIB) $(OLDIFT_LIB) -I$(HDR) -lm \
# 	-o ../../bin/crop_scn

# flip_scn:
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(SRC)/flip_scn.c \
# 	$(OPF_LIB) $(OLDIFT_LIB) -I$(HDR) -lm \
# 	-o ../../bin/flip_scn

# rotate_scn:
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(SRC)/rotate_scn.c \
# 	$(OPF_LIB) $(OLDIFT_LIB) -I$(HDR) -lm \
# 	-o ../../bin/rotate_scn

# rotate_scn2:
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) $(SRC)/rotate_scn2.c \
# 	$(OPF_LIB) $(OLDIFT_LIB) -I$(HDR) -lm \
# 	-o ../../bin/rotate_scn2

# thalamus:
# 	$(CC) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) \
# 	$(SRC)/thalamus.c -I$(HDR) \
# 	-L. -lcode_c $(OPF_LIB) $(OLDIFT_LIB) -lm -pthread \
# 	-o ../../bin/thalamus_bin

clean:
	$(RM) $(LIB)/lib*.a *~ *.o $(OBJ)/*.o -f

