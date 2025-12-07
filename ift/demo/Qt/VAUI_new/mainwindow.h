#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCore>
#include <QtGui>
#include <QGraphicsScene>
#include <QGraphicsEllipseItem>
#include <QListWidgetItem>
#include <QTreeWidgetItem>
#include <QProgressBar>
#include <QProgressDialog>

#include "ift.h"
#include "mynode.h"
#include "qcustomplot.h"
#include "mygraphicwidget.h"
#include "mywidgetitem.h"
#include "mylistwidget.h"
#include "dialogsceneoptions.h"


namespace Ui {
class MainWindow;
}

//#include "mythread.h"
class MyThread ;



class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    //variable
    QHash<int, QColor> hashClusterId2ClusterColor;
    QHash<int, QString> hashClusterId2ClusterName;
    QHash<int, QColor> hashLabelId2LabelColor;
    QHash<int, QString> hashLabelId2LabelName;


    QHash<unsigned long, QTreeWidgetItem*> hashSampleId2TreeItem;
    QHash<QTreeWidgetItem*, unsigned long> hashTreeItem2SampleId;

    QHash<unsigned long, MyNode*> hashSampleId2GraphNodeRef;
    QHash<MyNode*,unsigned long> hashGraphNodeRef2SampleId;

    QHash<unsigned long, unsigned long> hashSampleId2DatasetRow;

    QHash<unsigned long, QString> hashSampleId2Path;
    QHash<QString,unsigned long> hashPath2SampleId;

    QHash<unsigned long, MyWidgetItem*> hashSampleId2CurrentItemInList;
    QHash<unsigned long, bool> hashSampleId2CurrentSelected;

    QVector<QString> labelsNames;
    QVector<QColor> labelsColor;

    QList<QString> nodeColorOptions;
    QList<QString> binaryOptions;
    QProgressDialog* progressDialog = NULL;
    QProgressBar* bar = NULL;
    iftTsne* tsne = NULL;
    iftCommonActiveLearning* activeLearning = NULL;
    iftDataSet* workDataset = NULL;
    iftDataSet* testDataset = NULL;
    iftDataSet* datasetLowDimension = NULL;
    MyThread* thread = NULL;
    QString directoryOfImages = "/home/deangeli/base_parasites/Base_+4/Protozoa/orig";
    int countTest = 0;

    QVector<int> selectedSamplesIndicesVec;
    QVector< QVector<int> > selectedSamplesIndicesVecHistory;
    QVector<int> numberOfSupervisedSamplesHistory;
    QVector<int> numberOfLabelPropagatedSamplesHistory;

    QVector<float> accuracy;
    QVector<float> accuracyNorm;
    QVector<float> cohenKappa;
    QVector<int> learnCycleValues;
    QVector<MyNode *>graphNodes;
    QVector<MyEdge *>graphEdges;
    QString currentNodeColorOption;
    QString currentEdgeOption;
    QString currentBinaryOption;

    bool runningProjection = false;
    QCustomPlot* plot = NULL;
    int currentLabel;
    int filterCluster = 0;
    int filterClass = 0;
    int sceneScalingFactor = 0;
    bool manualSelection = false;
    int marginOscilator_forBugCorrection = 1;
    MyListWidget* mylist = NULL;
    QVector<int> allSupervised;

    QTreeWidget* treeWidgetSamples = NULL;
    MyGraphicWidget* myGraphicWidgetProjection = NULL;
    QGraphicsScene* scene = NULL;
    iftFileSet* fileSet = NULL;
    iftActiveLearningState previousState = ACTIVELEARNING_BEGIN;

    //Selectors
    iftSelectorRandom* selectorRandom;
    iftSelectorRDS* selectorRDS;
    iftSelectorRWS* selectorRWS;
    iftGenericSelector* selector;

    //classifiers
    iftGenericClassifier *classifier;
    iftMatrix* confusionMatrixNormalized = NULL;
    iftMatrix* confusionMatrixCount = NULL;
    iftMatrix* confusionMatrixSmallDatasetMean = NULL;
    iftMatrix* confusionMatrixSmallDatasetStandardDeviation = NULL;

    float meanAcc;
    float stdevAcc;
    float meanAccNorm;
    float stdevAccNorm;
    float meanKappa;
    float stdevKappa;

    QVector<float> accSmallDataset;
    QVector<float> accStdSmallDataset;
    QVector<float> accNormSmallDataset;
    QVector<float> accNormStdSmallDataset;
    QVector<float> kappaSmallDataset;
    QVector<float> kappaStdSmallDataset;

    QVector<float> accTrueLabelVec;
    QVector<float> accNormTrueLabelVec;
    QVector<float> kappaTrueLabelVec;

    float trueLabelAcc;
    float trueLabelAccNorm;
    float trueLabelKappa;
    bool seeTrueLabelInfo = false;
    QVector<int>operationNumber;
    int nSupervisedSamples;
    int nPropagatedLabelSamples;

    bool developingMode = false;

    DialogSceneOptions* windowSceneOptions = NULL;

    //functions
    void addRoot(QString name,QString description);
    void addChild(QTreeWidgetItem* parent, QString name,QString description);
    void openDataSetAndUpdateData();
    void clearAllData();
    void clearAllHashTable();
    void clearAllDatasets();
    void clearActiveLearning();
    void resetDataSetLabels(iftDataSet* dataset);
    void createDefaultClasses(iftDataSet* dataset);
    void initProgressBar();
    void creatIconsOnThumb(iftDataSet* dataset);
    QString mountSampleName(iftSample* sample);
    void initActiveLearningManager(iftDataSet* dataset);
    void createColorTableForClusters(iftDataSet* dataset);
    void createTreeViewForOrganization(iftDataSet*  dataset,
                                       iftGenericVector* clusterOrganization,
                                       QTreeWidget* treeWidget);
    void createGraphNodes(iftDataSet *dataset2D);
    void createGraphNodes(iftDataSet *dataset2D, MainWindow* mainwindow);
    void updateImages(iftDataSet* dataset);
    void initScene();
    void initColorNodeOptionsList();
    void initInfoArea();
    void initPlot();
    void initGraphicsView();
    void paintNodeSupervisionedClass(iftDataSet* dataset);
    void paintNodeClass(iftDataSet* dataset);
    void paintNodeCluster(iftDataSet* dataset);
    void paintNodePredictedClass(iftDataSet* dataset);
    void paintNodeTextClusterSup(iftDataSet* dataset);
    void paintNodeTextClusterFrontier(iftKnnGraph* graph);
    void createRepresentativeSamplesElementsInList(iftDataSet* dataset);
    void createVectorWithLabelsName2FillComboBox();
    void createVectorWithLabelsColor2FillComboBox();
    void paintNode(QString comboBoxOption);
    void highlightNode(MyNode* node);
    void unhighlighNode(MyNode* node);
    void displaySelectedSamples(int start, int end);
    bool checkIfAllSamplesWereSupervised();
    void findIconsForClassesFacebook();
    void setUpSelectedSamples2Train();
    void setTrainAndTestSamples();
    void updateInfoArea();
    void go2NextAciveLeaningState();
    void go2PreviousAciveLeaningState();
    void updatePlot();
    void printDatasetInfo(iftDataSet* dataset);
    void unfindIconsForClassesFacebook();
    void unsetUpSelectedSamples2Train();
    void createNamesForClusters(iftDataSet* dataset);
    void updateComboBoxPropgation();
    void propagateLabels(int label);
    void setStatusSupervisedandUnsupervisedSamples(iftDataSet* dataset,iftSampleStatus statusSupervised,iftSampleStatus statusUnsupervised);
    void highlightSelectedSamples();
    void undisplaySelectedSamplesF();
    void unhighlightSelectedSamples();
    void updateSamplesInDatasetToTrain();
    void addNewRegisterInHistoryVectors();
    void updateNumberOfSupervisedSamples();
    void unupdateSamplesInDatasetToTrain();
    void unupdateNumberOfSupervisedSamples();
    void removeNewRegisterInHistoryVectors();
    void ungetSamplesIndexRDS();
    void createMapBetweenRowsAndIds(iftDataSet* dataset);
    void selectSamplesManually();
    void paintNodeBaseOnDistance2CentroidCluster(iftDataSet* dataset,
                                                 iftGenericVector* clusterOfSamplesIndex,
                                                 float** clusterDistance);
    void updateTreeItemsToolTips(iftDataSet* dataset);
    void selectSamplesAutomatically();
    void updateDrawingOptionForNodes(QString option);
    void initSelectors(iftDataSet* dataset);
    void computeCrossValidationOnSupervisedSamples();
    void updateTestDataset();
    void updateComboBoxFilter();
    void paintNodeTextClusterAll(iftDataSet* dataset);
    void paintNodeTextClusterHeatMap_distance(iftDataSet* dataset,iftGenericVector* clusterOfSamplesIndex,float** clusterDistance);
    void setDrawEdges(const QString &arg1);
    void drawPredecessorEdges(iftKnnGraph* graph);
    void drawNoneEdges(iftDataSet* dataset);
    void drawNeighbourEdges(iftKnnGraph* graph);
    void paintNodeTextClusterHeatMap_geodesicDistance(iftKnnGraph* graph);
    void paintNodeTextHiddenNonFrontier();
    void setDrawNodeBinary(const QString &arg1);
    void drawNodeBinaryNone(iftDataSet* dataset);
    void drawNodeBinaryFrontierCluster(iftKnnGraph* graph);
    void drawNodeBinaryRootMismatch(iftDataSet* dataset,iftGenericVector* clusterOfSamplesIndex);
    void openWindowSupervisedSamples();
    void loadTextlabels();
    void clearVectors();
    void clearLists();
    void mountConfusionMatrixWidget();
    void updateTableWidget();
    void reprojectVisualization();
    void paintNodeTextClusterHeatMap_geodesicDistanceNormalized(iftKnnGraph* graph);
    void paintNodeTextClusterHeatMap_inFrontier(iftKnnGraph* graph);
    void saveWorkDataset();
    void trainClassifier();
    void computeTrueLabelInfo();
    void initVectors();
    void paintNodeTextClusterSupPropagated(iftDataSet* dataset);
    void initiWindowSceneOptions();

public slots:
    //void threadFinished();
    void threadFinished(int taskId);
    void updateSelectedItems_Tree2Graph(QTreeWidgetItem *item, int column);
    void updateSelectedItems_Graph2Tree();
    void updateNodeColor_ComboBox2Graph(const QString &comboBoxRow);
    void updateNodeColor_ComboBox2Graph2(const QString &comboBoxRow);
    void updateDisplayOfSelectedSamples_removed(QVector<int> indices);
    void mylistDestroyed(QObject*);
    void windowSceneOptionsClosed();
    void windowNodeColorPaintChanged();
    void windowLabelPropagationChanged(int labelIdPropagated);
    void createMouseMenuForPropagation(QContextMenuEvent *event);
    void windowFilterClusterChanged(int clusterIndexFiltred);
    void windowFilterClassChanged(int classIndexFiltred);
    void windowObjectSizeChanged(double objectSize);
    void windowConstrastChanged(int contrast);
    void windowBrightChanged(int bright);
    void windowBinaryOptionChanged();
    void windowShapeOptionChanged(int shapeOption);
    void windowEdgeOptionChanged();
    void setNodesPosition();
    void windowManualReprojectChanged();

private slots:
    void on_actionOpen_Dataset_triggered();

    void on_actionNext_State_triggered();

    void on_actionPrevious_State_triggered();

    void on_actionMetrics_triggered();

    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_pushButton_3_clicked();

    void on_spinBox_valueChanged(int arg1);

    void on_pushButton_4_clicked();

    void on_actionactionLoadLabels_triggered();

    void on_pushButton_5_clicked();

    void on_actionSave_Dataset_triggered();

    void on_pushButton_6_clicked();

    void on_actionSave_triggered();

    void on_pushButton_7_clicked();

    void on_tableWidget_doubleClicked(const QModelIndex &index);

    void on_pushButton_9_clicked();

    void on_pushButton_8_clicked();

    void on_actionScene_Options_triggered();

private:
    Ui::MainWindow *ui;

};

enum ThreadTaskType
{
    THREAD_INIT_ACTIVELEANING,
    THREAD_INIT_TREEVIEW,
    THREAD_INIT_PROJECTION,
    THREAD_INIT_GRAPH,
    THREAD_PROPAGATION_LABEL,
    THREAD_SEE_ONLY
};

class MyThread : public QThread
{
    Q_OBJECT
public:
    MyThread(MainWindow* window, ThreadTaskType taskType);
    MainWindow* mainwindow;
    void run();
    ThreadTaskType taskType;

signals:
    void myFinished(int _taskType);

};


#endif // MAINWINDOW_H
