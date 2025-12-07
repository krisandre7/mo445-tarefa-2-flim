#ifndef DIALOGSCENEOPTIONS_H
#define DIALOGSCENEOPTIONS_H

#include <QDialog>
#include <QSpinBox>
#include <QComboBox>

namespace Ui {
class DialogSceneOptions;
}

class DialogSceneOptions : public QDialog
{
    Q_OBJECT

public:
    QString currentPaintNodeOption;
    QString currentBinaryOption;
    QString currentEdgeOption;
    int currentShapeOption;
    int currentLabelPropagationIndex;
    int currentFilterClusterIndex;
    int currentFilterClassIndex;
    double objectSize;
    int contrastValue;
    int BrightValue;
    bool sceneAutomaticReprojection;

    QList<QString> nodeColorOptions;
    QList<QString> edgeOptions;
    QList<QString> propagationOptions;
    QList<QString> binaryOptions;
    QList<QString> shapeOptions;
    QList<QString> filterClusterOptions;
    QList<QString> filterClassOptions;

    QSpinBox* objectSize_spinBox;

    bool developingMode;


    explicit DialogSceneOptions(QWidget *parent = 0, bool _developingMode = false);
    ~DialogSceneOptions();
    void fillStaticComboBoxOptions();
    void closeEvent(QCloseEvent *event);
    void fillPropagationComboBox(QHash<int,QString>hashLabelId2LabelName, QHash<int,QColor>hashLabelId2LabelColor);
    void fillClusterFilterComboBox(QHash<int,QString>hashClusterId2ClusterName, QHash<int,QColor>hashClusterId2ClusterColor);
    void fillClassFilterComboBox(QHash<int,QString>hashLabelId2LabelName, QHash<int,QColor>hashLabelId2LabelColor);

private slots:
    void on_comboBoxNodoColor_currentIndexChanged(const QString &arg1);
    void on_comboBoxPropagation_activated(int index);


    void on_comboBoxFilterCluster_activated(int index);

    void on_comboBoxFilterClass_activated(int index);

    void on_spinBoxObjectSize_editingFinished();

    void on_horizontalSliderContrast_valueChanged(int value);

    void on_horizontalSliderBright_valueChanged(int value);

    void on_comboBoxBinary_activated(const QString &arg1);

    void on_comboBoxObjectStyle_activated(const QString &arg1);

    void on_comboBoxEdgeColor_activated(const QString &arg1);

    void on_checkBoxAutomaticReprojection_clicked(bool checked);

    void on_pushButton_clicked();

signals:
//    void updateDirectoryAndPrefix(QString prefix, QString DirectoryPath);
//    void windowClosed(void);
    void nodeColorPaintChanged();
    void binaryOptionChanged();
    void edgeOptionChanged();
    void shapeOptionChanged(int);
    void labelPropagationChanged(int);
    void filterClusterChanged(int);
    void filterClassChanged(int);
    void objectSizeEditFinished(double);
    void constrastValueChanged(int);
    void brightValueChanged(int);
    void sceneAutoProjectionChanged(bool);
    void sceneManualReproject();
    void windowClosed();

private:
    Ui::DialogSceneOptions *ui;
};

#endif // DIALOGSCENEOPTIONS_H
