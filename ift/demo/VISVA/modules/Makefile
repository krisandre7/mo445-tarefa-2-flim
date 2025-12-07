all: libmodules

LIB=.
OBJ=./obj


SLICEVIEW_DIR = $(MODULE_DIR)/2Dview/sliceview
SLICEVIEW_INCLUDE = -I$(SLICEVIEW_DIR)/include

SURFACE_DIR = $(MODULE_DIR)/3Dview/surface
SURFACE_INCLUDE = -I$(SURFACE_DIR)/include

RAYLEAPING_DIR = $(MODULE_DIR)/3Dview/rayleapingbacksteps
RAYLEAPING_INCLUDE = -I$(RAYLEAPING_DIR)/include

CURVILINEAR_DIR = $(MODULE_DIR)/3Dview/curvilinear
CURVILINEAR_INCLUDE = -I$(CURVILINEAR_DIR)/include

INTERACTIVE_DIR = $(MODULE_DIR)/segmentation/interactive
INTERACTIVE_INCLUDE = -I$(INTERACTIVE_DIR)/include

DRIFT_DIR = $(MODULE_DIR)/segmentation/drift
DRIFT_INCLUDE = -I$(DRIFT_DIR)/include

OIFT_DIR = $(MODULE_DIR)/segmentation/oift
OIFT_INCLUDE = -I$(OIFT_DIR)/include

MANUAL_DIR = $(MODULE_DIR)/segmentation/manual
MANUAL_INCLUDE = -I$(MANUAL_DIR)/include

COMBINE_DIR = $(MODULE_DIR)/segmentation/combine
COMBINE_INCLUDE = -I$(COMBINE_DIR)/include

THRESHOLD_DIR = $(MODULE_DIR)/segmentation/threshold
THRESHOLD_INCLUDE = -I$(THRESHOLD_DIR)/include

CSF_DIR = $(MODULE_DIR)/segmentation/csf
CSF_INCLUDE = -I$(CSF_DIR)/include

VENTR_DIR = $(MODULE_DIR)/segmentation/ventricles
VENTR_INCLUDE = -I$(VENTR_DIR)/include

BRAIN_DIR = $(MODULE_DIR)/segmentation/brain
BRAIN_INCLUDE = -I$(BRAIN_DIR)/include

BASICSTATS_DIR = $(MODULE_DIR)/analysis/basicstats
BASICSTATS_INCLUDE = -I$(BASICSTATS_DIR)/include

SYMMETRY_DIR = $(MODULE_DIR)/analysis/symmetry
SYMMETRY_INCLUDE = -I$(SYMMETRY_DIR)/include

INTPROFILE_DIR = $(MODULE_DIR)/analysis/intprofile
INTPROFILE_INCLUDE = -I$(INTPROFILE_DIR)/include

INTERPOLATION_DIR = $(MODULE_DIR)/preprocessing/interpolation
INTERPOLATION_INCLUDE = -I$(INTERPOLATION_DIR)/include

SETORIENTATION_DIR = $(MODULE_DIR)/preprocessing/setorientation
SETORIENTATION_INCLUDE = -I$(SETORIENTATION_DIR)/include

MSP_DIR = $(MODULE_DIR)/preprocessing/msp
MSP_INCLUDE = -I$(MSP_DIR)/include

INHOMOGENEITY_DIR = $(MODULE_DIR)/preprocessing/inhomogeneity
INHOMOGENEITY_INCLUDE = -I$(INHOMOGENEITY_DIR)/include

ROI_DIR = $(MODULE_DIR)/preprocessing/roi
ROI_INCLUDE = -I$(ROI_DIR)/include

ENHANCEMENT_DIR = $(MODULE_DIR)/preprocessing/enhancement
ENHANCEMENT_INCLUDE = -I$(ENHANCEMENT_DIR)/include

RIGIDREG_DIR = $(MODULE_DIR)/registration/rigidreg
RIGIDREG_INCLUDE = -I$(RIGIDREG_DIR)/include

SELFREG_DIR = $(MODULE_DIR)/registration/selfreg
SELFREG_INCLUDE = -I$(SELFREG_DIR)/include

BRAINCLUSTER_DIR = $(MODULE_DIR)/segmentation/braincluster
BRAINCLUSTER_INCLUDE = -I$(BRAINCLUSTER_DIR)/include

CLOUDSBRAIN_DIR = $(MODULE_DIR)/segmentation/cloudsbrain
CLOUDSBRAIN_INCLUDE = -I$(CLOUDSBRAIN_DIR)/include

SEGEXTERNAL_DIR = $(MODULE_DIR)/segmentation/external
SEGEXTERNAL_INCLUDE = -I$(SEGEXTERNAL_DIR)/include


MODULE_INCLUDE = $(SLICEVIEW_INCLUDE) $(SURFACE_INCLUDE) $(RAYLEAPING_INCLUDE) \
$(CURVILINEAR_INCLUDE) $(INTERACTIVE_INCLUDE) $(DRIFT_INCLUDE) $(OIFT_INCLUDE) $(MANUAL_INCLUDE) $(COMBINE_INCLUDE) \
$(THRESHOLD_INCLUDE) $(BRAINCLUSTER_INCLUDE) \
$(CSF_INCLUDE) $(BRAIN_INCLUDE) $(VENTR_INCLUDE) $(BASICSTATS_INCLUDE) $(SYMMETRY_INCLUDE) $(INTPROFILE_INCLUDE) \
$(INHOMOGENEITY_INCLUDE) $(ROI_INCLUDE) $(ENHANCEMENT_INCLUDE) \
$(INTERPOLATION_INCLUDE) $(SETORIENTATION_INCLUDE) \
$(MSP_INCLUDE) $(RIGIDREG_INCLUDE) $(SELFREG_INCLUDE) $(CLOUDSBRAIN_INCLUDE) $(SEGEXTERNAL_INCLUDE) \
$(BASIC_INCLUDE) $(NEWIFT_INCLUDE)


#Rules
libmodules: $(LIB)/libmodules.a
	echo "libmodules.a built..."

$(LIB)/libmodules.a: \
$(OBJ)/register_modules.o \
libmodulesliceview \
libmodulesurface \
libmodulerayleaping \
libmodulecurvilinear \
libmoduleinteractive \
libmoduledrift \
libmoduleoift \
libmodulemanual \
libmodulecombine \
libmodulethreshold \
libmodulecsf \
libmoduleventr \
libmodulebrain \
libmodulebasicstats \
libmodulesymmetry \
libmoduleintprofile \
libmoduleinterpolation \
libmodulesetorientation \
libmoduleinhomogeneity \
libmoduleroi \
libmoduleenhancement \
libmodulemsp \
libmodulebraincluster \
libmodulerigidreg \
libmoduleselfreg \
libmodulecloudsbrain \
libmodsegexternal \
libmodulebasic
	ar x $(SLICEVIEW_DIR)/libmodulesliceview.a
	ar x $(SURFACE_DIR)/libmodulesurface.a
	ar x $(RAYLEAPING_DIR)/libmodulerayleaping.a
	ar x $(CURVILINEAR_DIR)/libmodulecurvilinear.a
	ar x $(INTERACTIVE_DIR)/libmoduleinteractive.a
	ar x $(DRIFT_DIR)/libmoduledrift.a
	ar x $(OIFT_DIR)/libmoduleoift.a
	ar x $(MANUAL_DIR)/libmodulemanual.a
	ar x $(COMBINE_DIR)/libmodulecombine.a
	ar x $(THRESHOLD_DIR)/libmodulethreshold.a
	ar x $(CSF_DIR)/libmodulecsf.a
	ar x $(VENTR_DIR)/libmoduleventr.a
	ar x $(BRAIN_DIR)/libmodulebrain.a
	ar x $(BASICSTATS_DIR)/libmodulebasicstats.a
	ar x $(SYMMETRY_DIR)/libmodulesymmetry.a
	ar x $(INTPROFILE_DIR)/libmoduleintprofile.a
	ar x $(INTERPOLATION_DIR)/libmoduleinterpolation.a
	ar x $(SETORIENTATION_DIR)/libmodulesetorientation.a
	ar x $(INHOMOGENEITY_DIR)/libmoduleinhomogeneity.a
	ar x $(ROI_DIR)/libmoduleroi.a
	ar x $(ENHANCEMENT_DIR)/libmoduleenhancement.a
	ar x $(MSP_DIR)/libmodulemsp.a
	ar x $(BRAINCLUSTER_DIR)/libmodulebraincluster.a
	ar x $(RIGIDREG_DIR)/libmodulerigidreg.a
	ar x $(SELFREG_DIR)/libmoduleselfreg.a
	ar x $(CLOUDSBRAIN_DIR)/libmodulecloudsbrain.a
	ar x $(SEGEXTERNAL_DIR)/libmodsegexternal.a
	ar x $(BASIC_DIR)/libmodulebasic.a
	mv *.o $(OBJ)
	ar csr $(LIB)/libmodules.a $(OBJ)/*.o


$(OBJ)/register_modules.o: register_modules.cpp
	$(CXX) $(CC_INCLUDE) $(OLDIFT_INCLUDE) $(CODE_C_INCLUDE) \
	$(CODE_CPP_INCLUDE) $(WXFLAGS) $(GUI_INCLUDE) \
	$(MODULE_INCLUDE) $(VISVA_INCLUDE) $(OPF_INCLUDE) \
	-c register_modules.cpp -o $(OBJ)/register_modules.o

libmodulebasic:
	$(MAKE) -C $(BASIC_DIR)

libmodulesliceview:
	$(MAKE) -C $(SLICEVIEW_DIR)

libmodulesurface:
	$(MAKE) -C $(SURFACE_DIR)

libmodulerayleaping:
	$(MAKE) -C $(RAYLEAPING_DIR)

libmodulecurvilinear:
	$(MAKE) -C $(CURVILINEAR_DIR)

libmoduleinteractive:
	$(MAKE) -C $(INTERACTIVE_DIR)

libmoduledrift:
	$(MAKE) -C $(DRIFT_DIR)

libmoduleoift:
	$(MAKE) -C $(OIFT_DIR)

libmodulemanual:
	$(MAKE) -C $(MANUAL_DIR)

libmodulecombine:
	$(MAKE) -C $(COMBINE_DIR)

libmodulethreshold:
	$(MAKE) -C $(THRESHOLD_DIR)

libmodulecsf:
	$(MAKE) -C $(CSF_DIR)

libmoduleventr:
	$(MAKE) -C $(VENTR_DIR)

libmodulebrain:
	$(MAKE) -C $(BRAIN_DIR)

libmodulebasicstats:
	$(MAKE) -C $(BASICSTATS_DIR)

libmodulesymmetry:
	$(MAKE) -C $(SYMMETRY_DIR)

libmoduleintprofile:
	$(MAKE) -C $(INTPROFILE_DIR)

libmoduleinterpolation:
	$(MAKE) -C $(INTERPOLATION_DIR)

libmodulesetorientation:
	$(MAKE) -C $(SETORIENTATION_DIR)

libmoduleinhomogeneity:
	$(MAKE) -C $(INHOMOGENEITY_DIR)

libmoduleroi:
	$(MAKE) -C $(ROI_DIR)

libmoduleenhancement:
	$(MAKE) -C $(ENHANCEMENT_DIR)

libmodulemsp:
	$(MAKE) -C $(MSP_DIR)

libmodulebraincluster:
	$(MAKE) -C $(BRAINCLUSTER_DIR)

libmodulerigidreg:
	$(MAKE) -C $(RIGIDREG_DIR)

libmoduleselfreg:
	$(MAKE) -C $(SELFREG_DIR)

libmodulecloudsbrain:
	$(MAKE) -C $(CLOUDSBRAIN_DIR)

libmodsegexternal:
	$(MAKE) -C $(SEGEXTERNAL_DIR)

clean:
	$(RM) *~ *.o $(LIB)/lib*.a $(OBJ)/*.o -f
	$(MAKE) clean -C $(BASIC_DIR)
	$(MAKE) clean -C $(SLICEVIEW_DIR)
	$(MAKE) clean -C $(SURFACE_DIR)
	$(MAKE) clean -C $(RAYLEAPING_DIR)
	$(MAKE) clean -C $(CURVILINEAR_DIR)
	$(MAKE) clean -C $(INTERACTIVE_DIR)
	$(MAKE) clean -C $(DRIFT_DIR)
	$(MAKE) clean -C $(OIFT_DIR)
	$(MAKE) clean -C $(MANUAL_DIR)
	$(MAKE) clean -C $(COMBINE_DIR)
	$(MAKE) clean -C $(THRESHOLD_DIR)
	$(MAKE) clean -C $(CSF_DIR)
	$(MAKE) clean -C $(VENTR_DIR)
	$(MAKE) clean -C $(BRAIN_DIR)
	$(MAKE) clean -C $(BASICSTATS_DIR)
	$(MAKE) clean -C $(SYMMETRY_DIR)
	$(MAKE) clean -C $(INTPROFILE_DIR)
	$(MAKE) clean -C $(INTERPOLATION_DIR)
	$(MAKE) clean -C $(SETORIENTATION_DIR)
	$(MAKE) clean -C $(MSP_DIR)
	$(MAKE) clean -C $(BRAINCLUSTER_DIR)
	$(MAKE) clean -C $(INHOMOGENEITY_DIR)
	$(MAKE) clean -C $(ROI_DIR)
	$(MAKE) clean -C $(ENHANCEMENT_DIR)
	$(MAKE) clean -C $(RIGIDREG_DIR)
	$(MAKE) clean -C $(SELFREG_DIR)
	$(MAKE) clean -C $(CLOUDSBRAIN_DIR)
	$(MAKE) clean -C $(SEGEXTERNAL_DIR)
