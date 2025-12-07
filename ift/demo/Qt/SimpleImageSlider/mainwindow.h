#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtCore>
#include <QtGui>
#include <QFileDialog>
#include <QMessageBox>
#include <string>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QThread>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    void loadImage(QString imgFilename, bool scaleImg);
    void updateImageInfo(bool scaleImg);

    QString imageFolder;
    QString workingDir;
    int currentImgId;
    QString currentImgFilename;
    QStringList imgList;
    int graphicsViewInitialWidth;
    int graphicsViewInitialHeight;
    bool automaticExecution;
    QTimer *threadTimer = NULL;

private slots:
    void timer_slot();
    void on_actionOpen_folder_triggered();
    void on_horizontalSliderImageId_valueChanged(int value);
    void on_spinBoxImageId_valueChanged(int value);
    void on_checkBoxAutomaticExecution_stateChanged(int arg1);

private:
    Ui::MainWindow *ui;
    QGraphicsScene* graphicsScene;
};

#endif // MAINWINDOW_H
