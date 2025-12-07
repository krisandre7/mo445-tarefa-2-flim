#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "iftqtapi.h"

#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QSlider>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    m_ui(new Ui::MainWindow)
{
    m_ui->setupUi(this);

    createSignals();

    m_msskel = nullptr;
    m_dist = nullptr;

    m_loaded = false;
    m_threshold = getSliderPosition(m_ui->skelSlider->value());
    m_reconst = m_ui->reconstCheckBox->isChecked();
	
    m_thold_scene = new QGraphicsScene(m_ui->thresholdView);
    m_msskel_scene = new QGraphicsScene(m_ui->msskelView);

    m_ui->thresholdView->setScene(m_thold_scene);
    m_ui->msskelView->setScene(m_msskel_scene);
}


MainWindow::~MainWindow()
{
    delete m_ui;
    if (m_msskel) iftDestroyFImage(&m_msskel);
    if (m_dist) iftDestroyImage(&m_dist);
}


void MainWindow::loadImage()
{
    if (maybeSave()) {
        QString file_name = QFileDialog::getOpenFileName(this, tr("Open Image"), QDir::currentPath());
        if (!file_name.isEmpty()) {
            openImage(file_name);
        }
    }
}


void MainWindow::createSignals()
{
    connect(m_ui->actionOpen_Image, &QAction::triggered, this, &MainWindow::loadImage);
    connect(m_ui->actionSave_View, &QAction::triggered, this, &MainWindow::saveCanvas);
    connect(m_ui->skelSlider, &QSlider::valueChanged, this, &MainWindow::setThreshold);
    connect(m_ui->reconstCheckBox, &QCheckBox::toggled, this, &MainWindow::setReconst);
}


bool MainWindow::maybeSave()
{
    if (m_loaded) {
       QMessageBox::StandardButton ret;
       ret = QMessageBox::warning(this, tr("skel-viz"),
                          tr("The skeleton has been computed.\n"
                             "Do you want to save it?"),
                          QMessageBox::Save | QMessageBox::Discard
                          | QMessageBox::Cancel);
        if (ret == QMessageBox::Save) {
            return saveCanvas();
        } else if (ret == QMessageBox::Cancel) {
            return false;
        }
    }
    return true;
}


bool MainWindow::saveCanvas()
{
    if (!m_loaded) return false;

	QString init_path = QDir::currentPath() + "untitled.png";

	QString file_name = QFileDialog::getSaveFileName(this, tr("Save As"),
			init_path, tr("Image (*.png, *.ppm, *.pgm, *.jpg, *.jpeg)"));

	if (file_name.isEmpty())
		return false;
	
	QImage image = (m_reconst) ? getReconst() : getSkel();
	image.save(file_name);

    return true;
}


bool MainWindow::openImage(const QString &file_name)
{
    QImage loaded_image;
    if (!loaded_image.load(file_name))
        return false;

    m_image = loaded_image.convertToFormat(QImage::Format_Grayscale8);
    m_ui->thresholdView->fitInView(m_image.rect(), Qt::KeepAspectRatio);
    m_ui->msskelView->fitInView(m_image.rect(), Qt::KeepAspectRatio);

    iftImage *bin_img = createSkel();

    m_loaded = true;

    drawMSSkel(bin_img);
    iftDestroyImage(&bin_img);

    drawToCanvas();

    return true;
}


iftImage *MainWindow::createSkel()
{
    if (m_msskel) iftDestroyFImage(&m_msskel);
    if (m_dist) iftDestroyImage(&m_dist);

    iftImage *bin_img = QImageToiftImage(m_image);
    if (!iftIsBinaryImage(bin_img)) {
        iftImage *aux = iftThreshold(bin_img, 127, 255, 255);
        iftDestroyImage(&bin_img);
        bin_img = aux;
    }

    iftAdjRel *A = iftCircular(static_cast<float>(sqrt(2.0)));
    m_msskel = iftMSSkel2D(bin_img, A, IFT_INTERIOR, &m_dist, nullptr);
    iftDestroyAdjRel(&A);

    return bin_img;
}


bool MainWindow::drawToCanvas()
{
    if (!m_loaded) return false;
	
    m_thold_scene->clear();
	QImage image = (m_reconst) ? getReconst() : getSkel();
    m_thold_scene->addPixmap(QPixmap::fromImage(image));

    return true;
}


void MainWindow::resizeEvent(QResizeEvent *)
{
    m_ui->thresholdView->fitInView(0, 0, m_thold_scene->width(),
                                   m_thold_scene->height(), Qt::KeepAspectRatio);

    m_ui->msskelView->fitInView(0, 0, m_msskel_scene->width(),
                                m_msskel_scene->height(), Qt::KeepAspectRatio);
}


QImage MainWindow::getSkel()
{
	if (!m_loaded)
		throw std::invalid_argument("Skeleton must be computed before thresholding.");

	iftImage *skel = iftFThreshold(m_msskel, m_threshold, 100.0, 1);
	iftImage *aux = iftBinaryFrom1To255(skel);
	iftDestroyImage(&skel);
	QImage image = iftImageToQImage(aux);
	iftDestroyImage(&aux);
	return image;
}


bool MainWindow::drawMSSkel(const iftImage *bin_image)
{
    if (!m_loaded) return false;

    m_msskel_scene->clear();
    QImage image = iftFImageToQImage(m_msskel);
    drawBorders(image, bin_image);
    m_msskel_scene->addPixmap(QPixmap::fromImage(image));

    return true;
}


QImage MainWindow::getReconst()
{
	if (!m_loaded)
		throw std::invalid_argument("Skeleton must be computed before reconstruction.");
    iftImage *skel = iftFThreshold(m_msskel, m_threshold, 100.0, 1);
    iftImage *axis = iftMult(skel, m_dist);
    iftDestroyImage(&skel);
    iftImage *rec = iftShapeReconstruction(axis, 255);
    iftDestroyImage(&axis);
    QImage image = iftImageToQImage(rec);
    iftDestroyImage(&rec);
    return image;
}


float MainWindow::getSliderPosition(int position)
{
    return position / 100.0f;
}


void MainWindow::setReconst(bool value)
{
    if (value == m_reconst) return;

    m_reconst = value;

    drawToCanvas();
}


void MainWindow::setThreshold(int value)
{
    float threshold = getSliderPosition(value);

    if (std::fabs(threshold - m_threshold) < // threshold == m_threshold
            std::numeric_limits<float>::epsilon()) return;

	m_threshold = threshold;

	drawToCanvas();
}


