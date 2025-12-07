#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QLabel>
#include "global.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    QImage *image2LabelArea(iftImage *image);

    void displayQimageOnLabel(QImage *image,QLabel *imageArea);

    void imageRendering(iftImage* image3D, float thetax_degree,
                             float thetay_degree,iftImage** outputImage);

    void computeBrightValue(iftMatrix *p0,iftMatrix *p1,iftMatrix *pn,
                            iftImage *image3D,iftImage *outputImage,
                            int pixelIndex, float cubeDiagonal,
                            int imageMaximumValue,float lambdaMin);

    iftMatrix *computeGradient(iftImage* labelImage);
    float computeDiagonal(float x,float y,float z);
    ~MainWindow();

private slots:
    void on_actionOpen_triggered();

    void on_doubleSpinBoxThetaX_valueChanged(double arg1);

    void on_doubleSpinBoxThetaY_valueChanged(double arg1);

    void clickEvent3dArea(const QPoint& pos);
    void pressedEvent3dArea(const QPoint& pos);
    void releasedEvent3dArea(const QPoint& pos);

private:
    Ui::MainWindow *ui;
    QImage *Qimage2D;
    QImage *Qimage3D;
};

#endif // MAINWINDOW_H
