#include "global.h"

iftDataSet *originalDataset = NULL;
iftDataSet *poolDataset = NULL;
iftDataSet *datasetTest = NULL;
iftDataSet *workDataset = NULL;
iftDataSet *poolLowDimension = NULL;
iftDataSet *datasetLowDimension = NULL;
iftKnnGraph *opfClusteringGrapgh  = NULL;

int mainWindowHeight = 0;
int mainWindowWidth = 0;
int mainWindowInitHeight = 0;
int mainWindowInitWidth = 0;

int sceneProjectionAreaHeight = 0;
int sceneProjectionAreaWidth = 0;
int sceneProjectionAreaX = 0;
int sceneProjectionAreaY = 0;

int scrollAreaHeight=0;
int scrollAreaWidth=0;
int scrollAreaX=0;
int scrollAreaY=0;

int sampleVerticalRadius = 1;
int sampleHorizontalRadius = 1;

int viewShift = 1;

int imageHeight = 0;
int imageWidth = 0;
int drawScaleHeightFactor = 0;
int drawScaleWidthFactor = 0;
int drawBorderDistance = 15;
int drawBorderDistanceNormalized = 0;
float rootSampleDiameter = 12;
float commonSampleDiameter = 6;
float alphaRootSample = 0.7;
float alphaCommonSample = 0.5;

bool* isRoot = NULL;
int edgeFactor = 0;
int activeIter = 1;
int nClusters = 0;
int nClasses = 0;

iftColorTable* colorTableCluster = NULL;
iftColorTable* colorTableClass = NULL;

iftSelectorFirstIterBehavior activeUSAFirstIter;
iftSelectorAlgorithm activeUSAlgorithm;
iftUnlabeledSelector * activeSelector = NULL;
iftActiveClassifier * activeClassifier = NULL;
iftClassifierAlgorithm classifierType = OPFSEMI_CLASSIFIER;

int numberTrainSamples;
int numberTestSamples;
bool shuffleDataset = true;
float datasetSplitFactor = 1.0;

float OPFUNSUP_kmaxPercentage = 0.05;
int  maximumNumberIterations = 10;
int stepMoment = 0;

int R_borderSelected = 0;
int G_borderSelected = 191;
int B_borderSelected = 255 ;

iftList * selectedSamples = NULL;
int numberSupervisedSamples = 0;
int currentLearningCycle = 0;
bool isCategoriesLoaded = false;
char erroMessagesBuffer[500];
ProgramStatus programStatus;
ObjectDrawingOption currentDrawingNodeOption = COLOR_TRUELABELBASED;
bool alreadyComputedLabelsForAllSamples = false;
//DrawingManager drawingManager;



