#ifndef GLOBALVARIABLES_H
#define GLOBALVARIABLES_H


#include "ift.h"
#include "activelearningutils.h"

//global variable to mainwindow
extern iftDataSet *originalDataset;
extern iftDataSet *artificialDataset;
extern iftDataSet* validationDataset;
extern iftDataSet *poolDataset;
extern iftDataSet *testDataset;
extern iftDataSet *noImpDataset;
extern iftDataSet *workDataset;
extern iftMatrix* preprocessing;
extern iftMatrix* ldaData;
extern iftDataSet *poolLowDimension;
extern iftDataSet *datasetLowDimension;
extern iftKnnGraph *opfClusteringGrapgh;

extern int mainWindowHeight;
extern int mainWindowWidth;
extern int mainWindowInitHeight;
extern int mainWindowInitWidth;

extern int sceneProjectionAreaHeight;
extern int sceneProjectionAreaWidth;
extern int sceneProjectionAreaX;
extern int sceneProjectionAreaY;

extern int scrollAreaHeight;
extern int scrollAreaWidth;
extern int scrollAreaX;
extern int scrollAreaY;

extern int sampleVerticalRadius;
extern int sampleHorizontalRadius;

extern int viewShift;

extern int imageHeight;
extern int imageWidth;
extern int drawScaleHeightFactor;
extern int drawScaleWidthFactor;
extern int drawBorderDistance;
extern int drawBorderDistanceNormalized;
extern float rootSampleDiameter;
extern float commonSampleDiameter;
extern float alphaRootSample;
extern float alphaCommonSample;

extern bool* isRoot;
extern int edgeFactor;
extern int activeIter;
extern int nClusters;
extern int nClasses;

extern iftSelectorFirstIterBehavior activeUSAFirstIter;
extern iftSelectorAlgorithm activeUSAlgorithm;
extern iftUnlabeledSelector * activeSelector;
extern iftActiveClassifier * activeClassifier;
extern iftClassifierAlgorithm classifierType;

extern int numberTrainSamples;
extern int numberTestSamples;
extern bool shuffleDataset;
extern float datasetSplitFactor;

extern iftColorTable* colorTableCluster;
extern iftColorTable* colorTableClass;
extern float OPFUNSUP_kmaxPercentage;
extern int  maximumNumberIterations;
extern int stepMoment;

extern int R_borderSelected;
extern int G_borderSelected;
extern int B_borderSelected;

extern iftList * selectedSamples;
extern int numberSupervisedSamples;
extern int currentLearningCycle;

//typedef enum {
//    COLOR_NONE = 0,
//    COLOR_TRUELABELBASED = 1,
//    COLOR_OPFCLUSTERBASED = 2
//} NodeColorOption;

//typedef enum {
//    HIDDEN_NONE = 0,
//    HIDDEN_ALL = 1,
//    HIDDEN_UNSUPERVISED = 2,
//    HIDDEN_SUPERVISED = 3
//} NodeHiddenOption;


typedef enum {
    NONE,
    COLOR_TRUELABELBASED,
    COLOR_LABELBASED,
    COLOR_OPFCLUSTERBASED,
    HIDDEN_ALL,
    HIDDEN_UNSUPERVISED,
    HIDDEN_SUPERVISED,
    LABEL_PROPAGATION_AREA
} ObjectDrawingOption;

typedef enum{
    COLOR,
    HIDDEN,
    LABEL_PROPAGATION
}DrawingOption;

typedef enum {
    BEGIN,
    SAMPLES_SELECTION_FIRST,
    SAMPLES_LABELING_FIRST,
    CLASSIFIER_TRAINING_FIRST,
    SAMPLES_SELECTION,
    SAMPLES_LABELING,
    CLASSIFIER_TRAINING,

}ProgramStatus;

struct DrawingManager{
    ObjectDrawingOption objectOption;
    DrawingOption drawOption;
};



extern bool isCategoriesLoaded;
extern char erroMessagesBuffer[500];
extern ProgramStatus programStatus;
extern ObjectDrawingOption currentDrawingNodeOption;
extern bool alreadyComputedLabelsForAllSamples;
//extern DrawingManager drawingManager;

#endif // GLOBALVARIABLES_H
