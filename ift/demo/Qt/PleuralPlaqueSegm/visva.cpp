#include "visva.h"
#include "ui_visva.h"
#include <QtWidgets>

Visva::Visva(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Visva)
{
    ui->setupUi(this);

    ui->saAxial->setBackgroundRole(QPalette::Dark);
    ui->saCoronal->setBackgroundRole(QPalette::Dark);
    ui->saSagital->setBackgroundRole(QPalette::Dark);

    createActions();
    createMenus();

    QWidget::showMaximized();
}

bool Visva::LoadFile(const QString &fileName)
{
    QImageReader reader(fileName);
    reader.setAutoTransform(true);
    const QImage image = reader.read();

    //Error when loading image
    if (image.isNull()) {
        QMessageBox::information(this, QGuiApplication::applicationDisplayName(),
                                 tr("Cannot load %1.").arg(QDir::toNativeSeparators(fileName)));
        setWindowFilePath(QString());
        ui->lblAxial->setPixmap(QPixmap());
        ui->lblAxial->adjustSize();
        return false;
    }

    imgLoad = image.copy();
    imgScaled = image.copy();

    ui->lblAxial->setPixmap(QPixmap::fromImage(imgLoad));

    scaleFactor = 1.0;

    zoomInAct->setEnabled(true);
    zoomOutAct->setEnabled(true);
    normalSizeAct->setEnabled(true);
    saveFileAct->setEnabled(true);

    ui->lblAxial->adjustSize();

    setWindowFilePath(fileName);
    return true;
}

void Visva::OpenFile()
{
    QStringList filters;
    filters << "Image files (*.png *.xpm *.jpg *.pgm)"
            << "Digital Imaging and Communications in Medicine (*.dicom)"
            << "Scene (*.scn)";
    const QStringList picturesLocations = QStandardPaths::standardLocations(QStandardPaths::PicturesLocation);
    QFileDialog dialog(this, tr("Open File"),
                       picturesLocations.isEmpty() ? QDir::currentPath() : picturesLocations.last());
    dialog.setAcceptMode(QFileDialog::AcceptOpen);
    dialog.setNameFilters(filters);

    while (dialog.exec() == QDialog::Accepted && !LoadFile(dialog.selectedFiles().first())) {}
}

void Visva::SaveFile()
{

}

void Visva::NewFile()
{

}

void Visva::ZoomIn()
{
    if (scaleFactor < 3.0)
        scaleImage(1.25);
}

void Visva::ZoomOut()
{
    if (scaleFactor > 0.2)
        scaleImage(0.8);
}

void Visva::NormalSize()
{
    scaleFactor = 1.0;
    scaleImage(1.0);
}

void Visva::scaleImage(double factor)
{
    scaleFactor *= factor;

    imgScaled = imgScaled.scaled(scaleFactor * imgLoad.size(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    ui->lblAxial->setPixmap(QPixmap::fromImage(imgScaled));
}

void Visva::About()
{
    QMessageBox::about(this, tr("About QtVisva"),
            tr("<p>The <b>Qt Visva</b> software allows the segmentation and analysis "
               "of thoracic images, gathered from computadorized tomography (CT). "
               "The focus of this software is the detection of abnormalities in human "
               "lungs.</p>"
               "<p>This software provides the user with state-of-the-art methods of medical "
               "image segmentation and analysis.</p>"
               "<p>Created by: Azael de Melo e Sousa. Master Student in State University of Campinas.</p>"));
}

void Visva::Brightness(int value)
{
    //QImage aux = imgScaled.copy();
    qBrightness(imgScaled,value);
    ui->lblAxial->setPixmap(QPixmap::fromImage(imgScaled));
}

void Visva::Contrast(double value)
{
    //QImage aux = imgScaled.copy();
    qContrast(imgScaled,value);
    ui->lblAxial->setPixmap(QPixmap::fromImage(imgScaled));
}

void Visva::OpenBrightContrWin()
{
    imgScaled = imgLoad.copy();
    scaleImage(1.0);
    brightcontrwin = new BrightContrWin(this);
    brightcontrwin->setModal(true);
    brightcontrwin->show();
    //The component in BrightContrWin needs to be accessed by other means than
    //calling ui form BrightContrWin class
    connect(brightcontrwin, SIGNAL(changeBrightness(int)), this, SLOT(Brightness(int)));
    connect(brightcontrwin, SIGNAL(changeContrast(double)), this, SLOT(Contrast(double)));
}

void Visva::createActions()
{
    newFileAct = new QAction(tr("&New"),this);
    newFileAct->setShortcut(tr("Ctrl+N"));
    connect(newFileAct,SIGNAL(triggered()),this,SLOT(NewFile()));

    openFileAct = new QAction(tr("&Open"),this);
    openFileAct->setShortcut(tr("Ctrl+O"));
    connect(openFileAct,SIGNAL(triggered()),this,SLOT(OpenFile()));

    saveFileAct = new QAction(tr("&Save"),this);
    saveFileAct->setShortcut(tr("Ctrl+S"));
    saveFileAct->setEnabled(false);
    connect(saveFileAct,SIGNAL(triggered()),this,SLOT(SaveFile()));

    exitAct = new QAction(tr("&Exit"),this);
    exitAct->setShortcut(tr("Ctrl+E"));
    connect(exitAct,SIGNAL(triggered()),this,SLOT(close()));

    zoomInAct = new QAction(tr("Zoom &In (25%)"),this);
    zoomInAct->setShortcut(tr("Ctrl++"));
    zoomInAct->setEnabled(false);
    connect(zoomInAct,SIGNAL(triggered()),this,SLOT(ZoomIn()));

    zoomOutAct = new QAction(tr("Zoom &Out (25%)"),this);
    zoomOutAct->setShortcut(tr("Ctrl+-"));
    zoomOutAct->setEnabled(false);
    connect(zoomOutAct,SIGNAL(triggered()),this,SLOT(ZoomOut()));

    normalSizeAct = new QAction(tr("Normal Size"),this);
    normalSizeAct->setShortcut(tr("Ctrl+M"));
    normalSizeAct->setEnabled(false);
    connect(normalSizeAct,SIGNAL(triggered()),this,SLOT(NormalSize()));

    aboutAct = new QAction(tr("&About"),this);
    aboutAct->setShortcut(tr("Ctrl+A"));
    connect(aboutAct,SIGNAL(triggered()),this,SLOT(About()));

    connect(ui->btOpenProject,SIGNAL(clicked(bool)), this, SLOT(OpenFile()));
    connect(ui->btZoomIn,SIGNAL(clicked(bool)), this, SLOT(ZoomIn()));
    connect(ui->btZoomOut,SIGNAL(clicked(bool)), this, SLOT(ZoomOut()));
    connect(ui->btBrightContr,SIGNAL(clicked(bool)), this, SLOT(OpenBrightContrWin()));
}

void Visva::createMenus()
{
    fileMenu = new QMenu(tr("&File"),this);
    fileMenu->addAction(newFileAct);
    fileMenu->addAction(openFileAct);
    fileMenu->addSeparator();
    fileMenu->addAction(saveFileAct);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAct);

    viewMenu = new QMenu(tr("&View"),this);
    viewMenu->addAction(zoomInAct);
    viewMenu->addAction(zoomOutAct);
    viewMenu->addAction(normalSizeAct);

    helpMenu = new QMenu(tr("&Help"),this);
    helpMenu->addAction(aboutAct);

    menuBar()->addMenu(fileMenu);
    menuBar()->addMenu(viewMenu);
    menuBar()->addMenu(helpMenu);
}

Visva::~Visva()
{
    delete ui;

    delete openFileAct;
    delete saveFileAct;
    delete newFileAct;
    delete exitAct;
    delete zoomInAct;
    delete zoomOutAct;
    delete normalSizeAct;
    delete aboutAct;

    delete fileMenu;
    delete viewMenu;
    delete helpMenu;
}
