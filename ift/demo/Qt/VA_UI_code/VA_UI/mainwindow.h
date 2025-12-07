#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>
#include <QPainter>
#include <QVBoxLayout>
#include <QtGui>
#include <QProgressBar>
#include <QProgressDialog>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsRectItem>
#include <QWheelEvent>
#include <QComboBox>
#include <QGroupBox>
#include "global.h"
#include "formdisplaysampleimage.h"
#include "dialogupdatesamplesimagedirectory.h"
#include "dialogloadingwindow.h"
#include "resizefilter.h"
#include "mygraphicscene.h"
#include "graphicsviewzoom.h"
#include "graphwidget.h"
#include "node.h"
#include "edge.h"
#include "activelearningutils.h"
#include "tsne.h"
#include "qcustomplot.h"


namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_listWidgetImagesThumb_doubleClicked(const QModelIndex &index);
    void updatePrefixAndDirectory(QString prefix,QString directory);
    void windowPrefixDirClosed();
    void updateSelectedSamples();
    void on_listWidgetImagesThumb_itemSelectionChanged();
    void mouseCustomSceneMenu(DrawingManager selectedAction);
    void on_pushButton_2_clicked();
    void updateColorOfCurrentSelectedSamples(const QString &row);
    void mouseMoveScene(QPointF* currentMousePositionInGraphicArea,
                        QPointF* currentMousePositionInScene);
    void on_actionDataSet_triggered();
    void on_actionCategories_2_triggered();
    void on_actionDataset_Images_Directory_2_triggered();
    void propagateLabel(int labelId);
    void on_actionSave_dataset_triggered();

private:
    //variables
    Ui::MainWindow *ui;
    QMessageBox messageBox;
    QHash<int, QColor> hashClusterId2ClusterColor;
    QHash<int, int> hashsampleId2labelId;
    QHash<int, QColor> hashLabelId2LabelColor;
    QHash<int, QString> hashLabelId2LabelName;
    QHash<int, QString> hashsampleId2FileLocation;

    QHash<unsigned long long, int> hashsampleId2GraphNodeIndex;

    QHash<QGraphicsItem*, int> hashNodeItemPointer2GraphNodeIndex;
    QHash<int, int> hashsampleId2IconIndex;
    QHash<int, bool> hashsampleId2SupervisedSample;
    QHash<int, QString> hashIconIndex2FileLocation;
    QString directory;
    QString prefix;
    FormDisplaySampleImage *formDisplaySampleImage;
    DialogUpdateSamplesImageDirectory *dialogUpdateSamplesImageDirectory;
    DialogLoadingWindow* dialogLoadingWindow;
    bool changes;
    GraphWidget *graphWidget;
    QList<Node *>graphNodes;
    QList<Edge *>graphPredMap;
    QList<Edge *>graphKnnMap;
    QList<QComboBox *>comboBoxes;
    QList<float>avaregeAccuracyOverLearningCycles;
    QList<float>standardDeviationOverLearningCycles;
    QColor colorForUnknowns;
    int* samplesPerCluster;
    DrawingManager drawingManager;
    QPointF mouseCoordInGraphicArea;
    QPointF mouseCoordInScene;
    int originalImageHeight;
    int originalImageWidth;
    int originalNchannels;
    iftCommonActiveLearning* activeLearning = NULL;
    QList<QString>listPaths2Images;
    iftDataSet* firstDataset;
    QList< QPair<int,QString> > listPairTrueLabelIdAndTrueLabelName;
    QList<iftList *>selectedSamplesHistory;


    QProgressDialog* progressDialog = NULL;
    QProgressBar* bar = NULL;
    iftTsne* tsne = NULL;


    //QVector<iftSample*>vectorSampĺe;
    //QVector<bool>vectorSampĺeSuprvised;

    QList<iftSample*>listSample;
    //QList<bool>listSampĺeSuprvised;
    bool resetTrueLabels;


    //functions

    //init
    void initComponents();
    void initMessageBox();
    void initGraphicsViewProjectionArea();
    void initListWidgetImagesThumb();
    void initLabelImageCurrentSample();
    void initTableWidgetSelectedSamples();
    void initGraphWidgets();
    void initTabWidget();
    void initWidgetPlot();
    void initLocalVariables();
    void initComponentsConnections();
    void initActiveLearningManager(iftDataSet* dataset);

    //dimensionality reduction
    void minMaxdataset(iftDataSet* dataset);
    iftDataSet* datasetDimensionalityReduction(iftDataSet *dataset);
    void resetDataSetLabels(iftDataSet* dataset);

    //graph
    void createGraphNodes(iftDataSet *dataset2D);
    void createGraphPredecessorEdgesPath(iftKnnGraph *opfClusteringGrapgh);
    void removeGraphNodes();
    void removeGraphPredEdgesPath();

    //drawing
    void highlightSelectedSamplesBySelector(iftList * selectedSamples);
    void drawNodes();
    void drawNodesNone();
    void drawNodesTrueLabel();
    void drawNodesCluster();
    void hiddenNodes();
    void hiddenNoneNodes();
    void hiddenAllNodes();
    void hiddenUnsupervisedNodes();
    void highlightNodeNone(int sampleIndex);
    void highlightNodeTrueLabel(int sampleIndex);
    void highlightNodeCluster(int sampleIndex);
    void unhighlightNodeNone(int sampleIndex);
    void drawNodesLabel();
    void unhighlightNodeTrueLabel(int sampleIndex);
    void unhighlightNodeCluster(int sampleIndex);
    void unhighlightSelectedSamplesBySelector(iftList * selectedSamples);
    void showPredecessorPath(Node *node);
    void showKnnNeighbours(Node *node);
    void hidePath(Node *node);
    void updateSampleColorForSuggestionLabel();
    void drawNodesInSceneFirstTime(bool paintNodes);

    //plot
    void createPlot();
    void updatePlot(float mean, float std, int iteration);

    //others
    void findSampleFile(QFileInfoList directoryList, QString fileName, int absolutePathTableIndex);
    void updateIconsOnThumb(iftDataSet *dataset,bool showProgressBar);
    void createContextMenuForProjectionArea(const QPoint &pos);
    //QString mountSampleName(int sampleId);
    QString mountSampleName(iftSample* sample);

    int* countNumberSamplesInClusters(iftKnnGraph *opfClusteringGrapgh,int nclusters);
    void getIconSelectedSamplesBySelector(iftList * selectedSamples);
    void prepareTable(iftList * selectedSamples);
    void updateHashColorAndName(QString listString);
    void prepareDatasetTest();
    float computeAccuracy();
    void validate();
    void setData();
    iftDataSet* loadDataSet(const char *pathname);
    void creatIconsOnThumb();
    void createColorTableForClusters(iftDataSet* dataset);
    void createWindowForIconInThumb(const QModelIndex &index);
    void createWindowToSetDirectory();
    void updateSceneFromThumb();
    void activeLearningMainFlow();
    bool updateHashLabelName(QString fileName_categories,bool insertUnknownCategorie = true);
    void updateHashLabelColor();
    void loadCategories();
    void saveCurrentDataSetLabels();
    void createDefaultClasses(iftDataSet* dataset);
    iftDataSet* createDatasetFromList(QList<iftSample*> list,int nfeats);
    void fillSampleVector();
    void iftResetTrueLabels(iftDataSet* dataset);
    void updateSampleIcon(QString imageFileAbsolutePath, int sampleId);
    void drawNodesInScene();
    void activeLearningFoward();
    void initProgressBar();
    void destroyProgressBar();

    //clear
    void clearAllData();
    void clearAllHashTable();
    void clearAllDatasets();
    void clearAllGraphNodes();
    void clearAllGraphEdges();
    void clearPlotArea();
    void clearThumbIcons();




protected:
    void resizeEvent(QResizeEvent* event);


signals:
    void currentProgress(int percentage);
};

#endif // MAINWINDOW_H
