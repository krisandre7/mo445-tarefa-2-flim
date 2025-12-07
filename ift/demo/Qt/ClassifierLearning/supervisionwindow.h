#ifndef SUPERVISIONWINDOW_H
#define SUPERVISIONWINDOW_H

#include <QDialog>
#include <QtCore>
#include <QtGui>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QMessageBox>

#include "ift.h"
//#include "mainwindow.h"

class MainWindow;

namespace Ui {
class SupervisionWindow;
}

class SupervisionWindow : public QDialog
{
    Q_OBJECT

public:
    explicit SupervisionWindow(QWidget *parent = 0, QString imgFilePath = "", int currentTrueLabel = 0, QHash<int, QColor> *hashLabelId2LabelColor = 0, QHash<int, QString> *hashLabelId2LabelName = 0);
    ~SupervisionWindow();
    void loadClassesToComboBox(QHash<int, QColor> hashLabelId2LabelColor, QHash<int, QString> hashLabelId2LabelName, int currentTrueLabel);
    void loadImage(QString imgFilePath);

    void enableWidgetsByMode(QString mode);

    void update3DImage();
    void update2DImage();

    int chosenClass;

private slots:
    void on_pushButtonSupervise_clicked();

    void on_pushButtonCancel_clicked();

    void on_pushButtonZoomIn_clicked();

    void on_pushButtonZoomOut_clicked();

    void on_pushButtonOriginalSize_clicked();

    void on_pushButtonFitWindow_clicked();

    void on_spinBoxSlice_valueChanged(int arg1);

    void on_comboBoxPlane_currentIndexChanged(const QString &arg1);

    void on_spinBoxBand_valueChanged(int arg1);

private:
    Ui::SupervisionWindow *ui;
    QGraphicsScene* scene;
    QImage *image = nullptr;

    // IFT Variables
    iftImage *img = nullptr;
    iftMImage *mimg = nullptr;

    // Qt Variables
    qreal abs_scaleFactor = 1.0;
    QString plane = "Axial";

};

#endif // SUPERVISIONWINDOW_H
