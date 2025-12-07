#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCore>
#include <QtGui>
#include <QFileDialog>
#include <QMessageBox>
#include <string>
#include <QThread>
#include <QColorDialog>
#include <QJsonObject>
#include <QJsonParseError>

#include "ift.h"
#include "globalvariables.h"
#include "graphnode.h"
#include "supervisionwindow.h"
#include "safetyindicator.h"

#include <iostream>
using namespace std;

namespace Ui {
class MainWindow;
}

typedef enum {
    THREAD_TSNE_PROJECTION,
    THREAD_OPF_CLASSIFICATION,
    THREAD_THUMBNAIL,
    THREAD_GRAPH_EDGES
} ThreadTaskType;

class MyThread;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    Ui::MainWindow *ui;

    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    /* datasets and graph */
    iftDataSet *originalDataSet = NULL;
    iftDataSet *workingDataSet = NULL;
    QVector<GraphNode *>graphNodes;
    QVector<GraphEdge *>graphEdges;
    QVector<GraphEdge *>KNNgraphEdges;
    QString workingDir;
    QString edgeMode;
    QString dataSetName;
    int numbSamplesShown;
    int numbSamplesSelected;
    SafetyIndicator *safetyInd = NULL;

    /* thread */
    MyThread* threadProjection = NULL;
    MyThread* threadThumbnail = NULL;
    MyThread* threadGraphEdges = NULL;
    bool runningProjection = false;
    ThreadTaskType threadTaskType;
    QTimer *threadTimer = NULL;
    QTime threadElapsedTime;

    /* hash structures */
    QHash<int, QColor> hashGroupId2GroupColor;
    QHash<int, QString> hashGroupId2GroupName;
    QHash<int, QColor> hashLabelId2LabelColor;
    QHash<int, QString> hashLabelId2LabelName;
    QHash<int, QColor> hashSampleId2GlobalkNNColor;
    QHash<int, QColor> hashSampleId2LocalkNNColor;

    //QHash<unsigned long, QTreeWidgetItem*> hashSampleId2TreeItem;
    //QHash<QTreeWidgetItem*, unsigned long> hashTreeItem2SampleId;

    QHash<unsigned long, GraphNode*> hashSampleId2GraphNodeRef;
    QHash<GraphNode*,unsigned long> hashGraphNodeRef2SampleId;

    QHash<unsigned long, QString> hashSampleId2Path;
    QHash<QString,unsigned long> hashPath2SampleId;

    //QHash<unsigned long, MyWidgetItem*> hashSampleId2CurrentItemInList;
    //QHash<unsigned long, bool> hashSampleId2CurrentSelected;

    /* graphics view */
    int sceneScalingFactor = 0;
    QGraphicsScene* projGraphicsScene = NULL;
    iftFileSet* fileSet = NULL;
    iftFileSet* thumbnailPath = NULL;
    QString currentSupervisionField;
    QString currentSupervisionMode;
    SupervisionWindow *supervisionWindow = NULL;
    QString currentNodeColorOption;
    bool drawShadowInPoints;
    float globalMinFeature = IFT_INFINITY_FLT;
    float globalMaxFeature = IFT_INFINITY_FLT_NEG;
    float globalMinFeatPerplexity = IFT_INFINITY_FLT;
    float globalMaxFeatPerplexity = IFT_INFINITY_FLT_NEG;
    float globalMinFeatMean = IFT_INFINITY_FLT;
    float globalMaxFeatMean = IFT_INFINITY_FLT_NEG;
    float globalMinFeatStdev= IFT_INFINITY_FLT;
    float globalMaxFeatStdev = IFT_INFINITY_FLT_NEG;
    float globalMinFeatWeight= IFT_INFINITY_FLT;
    float globalMaxFeatWeight = IFT_INFINITY_FLT_NEG;


    /* methods */
    void initScene();
    void initGraphicsView();
    void clearAllData(bool clearDatasets=true);
    void tsneProjectionOnCurrentDataSet();
    void createThumbnailOfCurrentDataSet();
    void enableWidgetsByMode(QString mode);
    void createHashBetweenClassesAndColors();
    void createHashBetweenGroupsAndColors();
    void createClassListView();
    void createGroupListView();
    void createSampleListView();
    QString formatTime(int miliseconds);
    DrawingOption nodeTypeToDrawOption(QString nodeType);
    bool resetPropagation();
    void createComboBoxForClassAndGroupFilters();
    void filterSamplesByClassGroupFeaturesAndNodeData();
    void setValuesForNodeDataFilterSpinBoxes();
    void updateNumbSamplesShown();
    void addNewClassForSupervision();
    void addNewGroupForSupervision();
    void addClassToSelectedClassesListView();
    void SelectGroupUp();
    void SelectGroupDown();
    void addGroupToSelectedGroupsListView();
    void addSampleToSelectedSamplesListView();
    void markSamplesAsSelectedByClassOrGroup();
    void updateDataSetInfoTextBoxes();
    void createGraphEdgesbykNN();
    
    void clearGraphEdges();
    double calcEntropy(QVector<int> freq, int n);
    void paintSafetyIndicator(GraphNode *node);
    void createHashBetweenIDAndSafetyIndicator();
    void createHashBetweenIDAndGlobalkNNColor();
    void createHashBetweenIDAndLocalkNNColor();

    void computeAutomaticPerplexity();
    void updateSampleTooltip();
    void updateLabelsSelectedClassesGroupsSamples();
    void verifyCheckBoxesInListViews();

private slots:
    void on_actionOpen_dataset_triggered();
    void on_horizontalSliderPerplexity_valueChanged(int value);
    void on_horizontalSliderNumIterTSNE_valueChanged(int value);
    void on_spinBoxPerplexity_valueChanged(int arg1);
    void on_spinBoxNumIterTSNE_valueChanged(int arg1);
    void on_pushButtonProjectTSNE_clicked();
    void on_pushButtonInvertSelection_clicked();
    void on_comboBoxNodeType_currentTextChanged();
    void on_comboBoxNodeColor_currentTextChanged();
    void on_comboBoxSupervisionField_currentTextChanged();
    void on_comboBoxSupervisionMode_currentTextChanged();
    void on_actionSave_dataset_triggered();
    void on_pushButtonResetSupervision_clicked();
    void on_comboBoxFilterClass_currentIndexChanged();
    void on_comboBoxFilterGroup_currentIndexChanged();
    void on_pushButtonSaveCanvas_clicked();
    void on_pushButtonChangeBkgColor_clicked();
    void on_comboBoxNodeDataFilter_currentTextChanged(const QString &arg1);
    void on_doubleSpinBoxNodeDataFilterFrom_valueChanged(double arg1);
    void on_doubleSpinBoxNodeDataFilterTo_valueChanged(double arg1);
    void on_checkBoxAutomaticPerplexity_stateChanged();
    void on_doubleSpinBoxFeaturesFrom_valueChanged(double arg1);
    void on_doubleSpinBoxFeaturesTo_valueChanged(double arg1);
    void on_checkBoxDisplayNodeDataFilter_stateChanged();
    void on_checkBoxInverseClassFilter_stateChanged();
    void on_checkBoxInverseGroupFilter_stateChanged();
    void on_pushButtonEnableShadow_clicked();
    void on_listViewSelectedClasses_clicked(const QModelIndex &index);
    void on_listViewSelectedGroups_clicked(const QModelIndex &index);
    void on_listViewSelectedSamples_clicked(const QModelIndex &index);
    void on_pushButtonSaveSelectedClasses_clicked();
    void on_pushButtonSaveSelectedGroups_clicked();
    void on_pushButtonSaveSelectedSamples_clicked();
    void on_pushButtonOpenSelectedClasses_clicked();
    void on_pushButtonOpenSelectedGroups_clicked();
    void on_pushButtonOpenSelectedSamples_clicked();
    void on_checkBoxShowClassGroupColorOnTooltip_stateChanged();

public slots:
    void projectionFinished_slot(bool tsneExecuted);
    void graphEdgesFinished_slot();
    void timer_slot();

};

/*////////////////////////////////// MYTHREAD //////////////////////////////////*/

class MyThread : public QThread
{
    Q_OBJECT
public:
    MyThread(MainWindow* window, ThreadTaskType taskType);
    MainWindow* mainWindow = NULL;
    ThreadTaskType taskType;
    void run();

signals:
    void projectionFinished_signal(bool tsneExecuted);
    void classificationFinished_signal();
    void thumbnailFinished_signal();
    void graphEdgesFinished_signal();
};

#endif // MAINWINDOW_H
