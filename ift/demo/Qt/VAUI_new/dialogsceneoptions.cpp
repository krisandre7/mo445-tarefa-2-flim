#include "dialogsceneoptions.h"
#include "ui_dialogsceneoptions.h"
#include <QDebug>

DialogSceneOptions::DialogSceneOptions(QWidget *parent, bool _developingMode) :
    QDialog(parent),
    ui(new Ui::DialogSceneOptions)
{
    ui->setupUi(this);
    developingMode = _developingMode;

    //node options
    if(developingMode == true){
        nodeColorOptions << "Class Color";
        nodeColorOptions << "Cluster Color";
    }
    nodeColorOptions << "Text cluster sup+propagated";
    //nodeColorOptions << "Text cluster sup";
    nodeColorOptions << "classifier prediction";
    nodeColorOptions << "Heatmap_distance";
    nodeColorOptions << "Heatmap_geodesic_distance";
    nodeColorOptions << "Heatmap_geodesic_distance_normalized";
    nodeColorOptions << "Heatmap_in_frontier";

    //edge options
    edgeOptions << "None";
    edgeOptions << "Predecessor_path";
    edgeOptions << "k-nearest";

    binaryOptions << "None";
    binaryOptions << "Cluster_frontier";
    binaryOptions << "Root_mismatch";

    shapeOptions << "Geometric shapes";
    shapeOptions << "Text";
    shapeOptions << "Image";

    fillStaticComboBoxOptions();

    objectSize_spinBox = ui->spinBoxObjectSize;
    currentPaintNodeOption = nodeColorOptions.at(0);
    sceneAutomaticReprojection = ui->checkBoxAutomaticReprojection->isChecked();

    //setWindowFlags(Qt::Window);
    //setWindowFlags(windowFlags() | Qt::WindowMinimizeButtonHint);

}

void DialogSceneOptions::fillStaticComboBoxOptions(){
    for (int i = 0; i < nodeColorOptions.size(); ++i) {
        ui->comboBoxNodoColor->addItem(nodeColorOptions.at(i));
    }

    for (int i = 0; i < edgeOptions.size(); ++i) {
        ui->comboBoxEdgeColor->addItem(edgeOptions.at(i));
    }

    for (int i = 0; i < binaryOptions.size(); ++i) {
        ui->comboBoxBinary->addItem(binaryOptions.at(i));
    }

    for (int i = 0; i < shapeOptions.size(); ++i) {
        ui->comboBoxObjectStyle->addItem(shapeOptions.at(i));
    }
    ui->comboBoxObjectStyle->setCurrentIndex(1);
}

void DialogSceneOptions::fillPropagationComboBox(QHash<int,QString>hashLabelId2LabelName, QHash<int,QColor>hashLabelId2LabelColor){
    ui->comboBoxPropagation->clear();
    ui->comboBoxPropagation->setCurrentIndex(0);
    QComboBox* comboBox = ui->comboBoxPropagation;
    int index = 0;
    for (int i = 0; i < hashLabelId2LabelName.size(); ++i) {
        comboBox->addItem(hashLabelId2LabelName[i],hashLabelId2LabelColor[i]);
        const QModelIndex idx = comboBox->model()->index(index++, 0);
        comboBox->model()->setData(idx,hashLabelId2LabelColor[i], Qt::DecorationRole);
    }
}

void DialogSceneOptions::fillClusterFilterComboBox(QHash<int,QString>hashClusterId2ClusterName, QHash<int,QColor>hashClusterId2ClusterColor){
    ui->comboBoxFilterCluster->clear();
    ui->comboBoxFilterCluster->setCurrentIndex(0);
    QComboBox* comboBox = ui->comboBoxFilterCluster;
    int index = 0;
    for (int i = 0; i < hashClusterId2ClusterName.size(); ++i) {
        comboBox->addItem(hashClusterId2ClusterName[i],hashClusterId2ClusterColor[i]);
        const QModelIndex idx = comboBox->model()->index(index++, 0);
        comboBox->model()->setData(idx,hashClusterId2ClusterColor[i], Qt::DecorationRole);
    }
}

void DialogSceneOptions::fillClassFilterComboBox(QHash<int,QString>hashLabelId2LabelName, QHash<int,QColor>hashLabelId2LabelColor){
    ui->comboBoxFilterClass->clear();
    ui->comboBoxFilterClass->setCurrentIndex(0);
    QComboBox* comboBox = ui->comboBoxFilterClass;
    int index = 0;
    for (int i = 0; i < hashLabelId2LabelName.size(); ++i) {
        comboBox->addItem(hashLabelId2LabelName[i],hashLabelId2LabelColor[i]);
        const QModelIndex idx = comboBox->model()->index(index++, 0);
        comboBox->model()->setData(idx,hashLabelId2LabelColor[i], Qt::DecorationRole);
    }

}


DialogSceneOptions::~DialogSceneOptions()
{
    delete ui;
}


void DialogSceneOptions::closeEvent(QCloseEvent *event){
    QDialog::closeEvent(event);
    emit windowClosed();
}

void DialogSceneOptions::on_comboBoxNodoColor_currentIndexChanged(const QString &arg1)
{
    currentPaintNodeOption = arg1;
    emit nodeColorPaintChanged();
}


void DialogSceneOptions::on_comboBoxPropagation_activated(int index)
{
    currentLabelPropagationIndex = index;
    emit labelPropagationChanged(index);
}

void DialogSceneOptions::on_comboBoxFilterCluster_activated(int index)
{
    currentFilterClusterIndex = index;
    emit filterClusterChanged(index);
}

void DialogSceneOptions::on_comboBoxFilterClass_activated(int index)
{
    currentFilterClassIndex = index;
    emit filterClassChanged(index);
}


void DialogSceneOptions::on_spinBoxObjectSize_editingFinished()
{
    objectSize = objectSize_spinBox->value();
    emit objectSizeEditFinished(objectSize);
}

void DialogSceneOptions::on_horizontalSliderContrast_valueChanged(int value)
{
    contrastValue = value;
    emit constrastValueChanged(value);
}

void DialogSceneOptions::on_horizontalSliderBright_valueChanged(int value)
{
    BrightValue = value;
    emit brightValueChanged(value);
}

void DialogSceneOptions::on_comboBoxBinary_activated(const QString &arg1)
{
    currentBinaryOption = arg1;
    emit binaryOptionChanged();
}

void DialogSceneOptions::on_comboBoxObjectStyle_activated(const QString &arg1)
{
    int shapeOption = 1;
    if(arg1 == "Geometric shapes"){
        shapeOption = 0;
    }else if(arg1 == "Text"){
        shapeOption = 1;
    }else if(arg1 == "Image"){
        shapeOption = 2;
    }
    emit shapeOptionChanged(shapeOption);
}

void DialogSceneOptions::on_comboBoxEdgeColor_activated(const QString &arg1)
{
    currentEdgeOption = arg1;
    emit edgeOptionChanged();
}

void DialogSceneOptions::on_checkBoxAutomaticReprojection_clicked(bool checked)
{
    sceneAutomaticReprojection = checked;
    emit sceneAutoProjectionChanged(checked);
}

void DialogSceneOptions::on_pushButton_clicked()
{
    emit sceneManualReproject();
}
