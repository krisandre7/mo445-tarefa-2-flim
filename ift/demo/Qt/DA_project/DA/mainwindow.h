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
#include "graphwidget.h"
#include "node.h"
#include "edge.h"
#include "activelearningutils.h"
#include "tsne.h"
#include "qcustomplot.h"
#include "formdisplaysampleimage.h"
#include "datasetviewer.h"
#include <QImage>
#include <QErrorMessage>

namespace Ui {
    class MainWindow;
}

class MainWindow : public QMainWindow {
Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);

    static QMessageBox *messageBox;

    ~MainWindow();

public slots:

    void updatePrefixAndDirectory(QString prefix, QString directory);

    void windowPrefixDirClosed();


    void mouseMoveScene(QPointF *currentMousePositionInGraphicArea,
                        QPointF *currentMousePositionInScene);


    void mouseClicked(QPointF *currentMousePositionInGraphicArea,
                      QPointF *currentMousePositionInScene, Qt::MouseButton buttons);

    void removeNodes();

    void datasetLoaded();

    iftMatrix *weightedAverageColumn(iftMatrix *matrix, float *weight);

    iftMatrix *weightedMedianColumn(iftMatrix *matrix, float *weight);

private:
    //variables

    Node *node;

    iftSampler *kfold;

    Ui::MainWindow *ui;
    QFuture<void> future;
    QHash<int, int> hashsampleId2labelId;
    QHash<int, QString> hashLabelId2LabelName;
    QHash<int, QString> hashsampleId2FileLocation;
    QHash<Node *, int> hashNodeItemPointer2sampleId;
    QHash<QGraphicsItem *, int> hashNodeItemPointer2GraphNodeIndex;
    QHash<int, int> hashsampleId2IconIndex;
    QHash<int, bool> hashsampleId2SupervisedSample;
    QHash<int, QString> hashIconIndex2FileLocation;
    QHash<int, QListWidgetItem *> hashSampleId2ListWidgetItem;

    QHash<int, iftSample *> hashSampleId2iftSample;
    QHash<int, iftSample *> hashSampleId2iftSamplePCA;
    QHash<int, int> hashSampleId2Kused;

    QString directory;
    QString prefix;
    QString lastDialogPath;
    DialogUpdateSamplesImageDirectory *dialogUpdateSamplesImageDirectory;
    bool changes;
    DataSetViewer *graphWidget;
    QList<Edge *> graphKnnMap;
    QList<float> avaregeAccuracyOverLearningCycles;
    QList<float> standardDeviationOverLearningCycles;
    QList<QString> listPaths2Images;


    QList<iftSample *> listSample;
    QList<int> listSampleKused;

    DrawingManager drawingManager;
    QPointF mouseCoordInGraphicArea;
    QPointF mouseCoordInScene;
    int currentNumberSamples;
    int lastId;
    iftMatrix *S;
    iftMatrix *V;
    iftMatrix *preProcessing;
    iftDataSet *datasetPreProcessing = NULL;

    int knearest;
    double pca_nComponents;
    bool applyPCA;
    iftMatrix* Vt;
    bool applyLDA;
    int originalImageHeight;
    int originalImageWidth;
    int originalNchannels;
    bool isSampleAnImage;

    //functions
    //init
    void initComponents();

    void initMessageBox();

    void initGraphicsViewProjectionArea();

    void initListWidgetImagesThumb();

    void initLabelImageCurrentSample();

    void initGraphWidgets();

    void initTabWidget();

    void initWidgetPlot();

    void initLocalVariables();

    void initComponentsConnections();

    iftSample *interpolateFeatures(iftSample newPoint);

    void updateIconsOnThumb(iftDataSet *dataset, bool showProgressBar);
    QString mountSampleName(iftSample *sample);
    void validate();

    void updateData();

    void openDataSet();

    bool updateDataSetInfo(QString pathname);

    void creatIconsOnThumb();

    void createWindowForIconInThumb(const QModelIndex &index);

    void createWindowToSetDirectory();

    void updateSceneFromThumb();

    bool updateHashLabelName(QString fileName_categories);

    void loadCategories();

    void updateSampleIcon(QString imageFileAbsolutePath, int sampleId);

    iftMatrix *applyPCAOnDataset(iftDataSet *dataset, double nComponents);

    iftMatrix *applyLDAOnDataset(iftDataSet *dataset);

    void updateKNearestSpinBox();

    void updateKnearest();

    void updatePCA_nComponents();

    void fillSampleHash();

    void fillSample();

    void changeSampleTrueLabel(int newTrueLabel);

    void trainClassifier();

    //clear
    void clearAllData();

    void clearAllHashTable();

    void clearAllDatasets();

    void clearAllGraphNodes();

    void clearAllGraphEdges();

    void clearThumbIcons();

    void saveDataset();


protected:
    void resizeEvent(QResizeEvent *event);


signals:

    void currentProgress(int percentage);

private slots:

    void on_spinBox_valueChanged(int arg1);

    void on_pca_componentsDoubleSpinBox_valueChanged(double arg1);

    void on_kNearestSpinBox_valueChanged(int arg1);

    void on_pushButton_2_clicked();

    void on_actionDataset_Test_triggered();

    void on_actionDataset_Validation_triggered();

    void on_actionDataSet_triggered();

    void on_pushButton_3_clicked();

    void on_preProcessComboBox_currentIndexChanged(int index);

    void on_pushButton_clicked();
    void on_actionSave_dataset_triggered();
};

#endif // MAINWINDOW_H
