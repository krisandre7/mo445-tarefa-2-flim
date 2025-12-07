#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    graphicsScene = new QGraphicsScene();
    ui->graphicsView->setScene(graphicsScene);
    graphicsViewInitialWidth = ui->graphicsView->geometry().width();
    graphicsViewInitialHeight = ui->graphicsView->geometry().height();

    threadTimer = new QTimer(this);
    connect(threadTimer, SIGNAL(timeout()), this, SLOT(timer_slot()));

    /* read last used imgDir */
    QFile imgDirFilePtr("lastWorkingDir.txt");
    if (imgDirFilePtr.open(QIODevice::ReadOnly)) {
        QTextStream stream(&imgDirFilePtr);
        QString line = stream.readLine();

        if(!line.isEmpty()) {
            if(QDir(line).exists())
                workingDir = line;
            else
                workingDir = QDir::currentPath();
        } else {
            workingDir = QDir::currentPath();
        }
        imgDirFilePtr.close();
    } else {
        workingDir = QDir::currentPath();
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_actionOpen_folder_triggered()
{
    imageFolder = QFileDialog::getExistingDirectory(this, tr("Open image folder"), workingDir);

    if (imageFolder.isEmpty())
        return;
    else {
        QDir directory(imageFolder);
        imgList = directory.entryList(QStringList() << "*.png" << "*.PNG", QDir::Files);

        if(imgList.size() > 0) {
            currentImgId = 1;
            currentImgFilename = imgList.at(currentImgId-1);
            ui->horizontalSliderImageId->setMaximum(imgList.size());
            ui->spinBoxImageId->setMaximum(imgList.size());
            updateImageInfo(true);
        }

        /* save the working directory */
        workingDir = QFileInfo(imageFolder).absolutePath();
        QFile workingDirFilePtr("lastWorkingDir.txt");
        if (workingDirFilePtr.open(QIODevice::WriteOnly)) {
            QTextStream stream(&workingDirFilePtr);
            stream << workingDir;
            workingDirFilePtr.close();
        }
        ui->checkBoxAutomaticExecution->setChecked(false);
    }

}

void MainWindow::loadImage(QString imgFilename, bool scaleImg)
{
    graphicsScene->clear();
    graphicsScene->update();

    QImage image(imgFilename);
    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(image));
    graphicsScene->addItem(item);

    if(scaleImg) {
        //graphicsScene->setSceneRect(0, 0, graphicsViewInitialWidth, graphicsViewInitialHeight);

        qreal width_ratio = (float)ui->graphicsView->geometry().width() / (float)image.width();
        qreal height_ratio = (float)ui->graphicsView->geometry().height() / (float)image.height();

        printf("ui->graphicsView->geometry().width(): %d, image.width(): %d\n", ui->graphicsView->geometry().width(), image.width());
        printf("ui->graphicsView->geometry().height(): %d, image.height(): %d\n", ui->graphicsView->geometry().height(), image.height());
        printf("width_ratio: %f, height_ratio: %f\n", width_ratio, height_ratio);
        fflush(stdout);

        qreal scale_factor;
        if(width_ratio <= height_ratio)
            scale_factor = width_ratio;
        else
            scale_factor = height_ratio;

        ui->graphicsView->resetMatrix();
        ui->graphicsView->scale(scale_factor, scale_factor);
    }

    graphicsScene->update();
    ui->graphicsView->show();
}

void MainWindow::updateImageInfo(bool scaleImg)
{
    QDir directory(imageFolder);
    loadImage(directory.absoluteFilePath(currentImgFilename), scaleImg);
    ui->labelImageName->setText(currentImgFilename);
    ui->labelDirPath->setText(directory.absolutePath());
    if(ui->horizontalSliderImageId->value() != currentImgId)
        ui->horizontalSliderImageId->setValue(currentImgId);
    if(ui->spinBoxImageId->value() != currentImgId)
        ui->spinBoxImageId->setValue(currentImgId);
}

void MainWindow::on_horizontalSliderImageId_valueChanged(int value)
{
    currentImgId = value;
    currentImgFilename = imgList.at(currentImgId-1);
    updateImageInfo(false);
}

void MainWindow::on_spinBoxImageId_valueChanged(int value)
{
    currentImgId = value;
    currentImgFilename = imgList.at(currentImgId-1);
    updateImageInfo(false);
}

void MainWindow::on_checkBoxAutomaticExecution_stateChanged(int arg1)
{
    automaticExecution = ui->checkBoxAutomaticExecution->isChecked();

    if(automaticExecution)
        threadTimer->start(100);
    else
        threadTimer->stop();
}

void MainWindow::timer_slot()
{
    if(currentImgId < imgList.size()) {
        currentImgId++;
        currentImgFilename = imgList.at(currentImgId-1);
        updateImageInfo(false);
        threadTimer->start(ui->doubleSpinBoxAutoExecSpeed->value()*1000);
    }
    else {
        threadTimer->stop();
    }

}
