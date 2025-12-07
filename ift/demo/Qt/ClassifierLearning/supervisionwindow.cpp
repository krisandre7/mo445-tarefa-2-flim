#include "supervisionwindow.h"
#include "ui_supervisionwindow.h"

SupervisionWindow::SupervisionWindow(QWidget *parent, QString imgFilePath, int currentTrueLabel, QHash<int, QColor> *hashLabelId2LabelColor, QHash<int, QString> *hashLabelId2LabelName) :
    QDialog(parent),
    ui(new Ui::SupervisionWindow)
{
    qDebug();

    ui->setupUi(this);

    scene = new QGraphicsScene();
    ui->graphicsView->setScene(scene);

    if(!imgFilePath.isEmpty())
        loadImage(imgFilePath);
    loadClassesToComboBox(*hashLabelId2LabelColor, *hashLabelId2LabelName, currentTrueLabel);
}

SupervisionWindow::~SupervisionWindow()
{
    delete ui;
    iftDestroyImage(&img);
    iftDestroyMImage(&mimg);
}

QImage *iftImageToQImage(iftImage *img)
{
    QImage *dst = new QImage(img->xsize,img->ysize,QImage::Format_RGB888);
    QColor color;

    if ((img->Cb == NULL) || (img->Cr == NULL)){
        #pragma omp parallel for private(color)
        for (int i=0;i<img->n;i++){
            color.setRed(img->val[i]);
            color.setGreen(img->val[i]);
            color.setBlue(img->val[i]);
            iftVoxel u = iftGetVoxelCoord(img,i);
            dst->setPixel(u.x,u.y,color.rgb());
        }
    } else {
        #pragma omp parallel for private(color)
        for (int i=0;i<img->n;i++){
            iftColor cin,cout;
            cin.val[0] = img->val[i];
            cin.val[1] = img->Cb[i];
            cin.val[2] = img->Cr[i];
            cout = iftYCbCrtoRGB(cin,255);
            color.setRed(cout.val[0]);
            color.setGreen(cout.val[1]);
            color.setBlue(cout.val[2]);
            iftVoxel u = iftGetVoxelCoord(img,i);
            dst->setPixel(u.x,u.y,color.rgb());
        }
    }

    return dst;
}

void SupervisionWindow::loadImage(QString imgFilename)
{
    qDebug();

    const char *a = iftFileExt(imgFilename.toUtf8().data());
    char extension[10];
    strcpy(extension,a);

    if (img != NULL)
        iftDestroyImage(&img);

    // reading image
    if (iftIsValidFormat(imgFilename.toUtf8().data())){
        if (iftCompareStrings(extension, ".mimg")){
            mimg = iftReadMImage(imgFilename.toUtf8().data());
            ui->spinBoxBand->setMaximum(mimg->m-1);
            ui->spinBoxBand->setValue(mimg->m/2);
            img = iftMImageToImage(mimg,255,mimg->m/2);
        } else {
            img = iftReadImageByExt(imgFilename.toUtf8().data());
        }
    } else {
        printf("Error when loading image. Format not known: %s.\nSupervisionWindow::loadImage",extension);
        fflush(stdout);
        exit(1);
    }

    //normalizing img
    iftImage *aux = iftNormalize(img,0,255);
    iftDestroyImage(&img);
    img = aux;
    aux = nullptr;

    //checking if it is 3D or 2D
    if (iftIs3DImage(img)){
        iftImage *slice = iftGetXYSlice(img,img->zsize/2);
        ui->spinBoxSlice->setValue(img->zsize/2);
        ui->spinBoxSlice->setMaximum(img->zsize-1);
        ui->spinBoxSlice->setMinimum(0);
        image = iftImageToQImage(slice);
        iftDestroyImage(&slice);
        if (iftCompareStrings(extension, ".mimg")){
            enableWidgetsByMode("MultBand3DImage");
        } else
            enableWidgetsByMode("3DImage");
    } else {
        image = iftImageToQImage(img);
        if (iftCompareStrings(extension, ".mimg")){
            enableWidgetsByMode("MultBand2DImage");
        } else
            enableWidgetsByMode("2DImage");
    }

    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*image));
    scene->addItem(item);


    if ((image->width() > ui->graphicsView->geometry().width()) || (image->height() > ui->graphicsView->geometry().height())) {
        qreal width_ratio = (float)ui->graphicsView->geometry().width() / (float)image->width();
        qreal height_ratio = (float)ui->graphicsView->geometry().height() / (float)image->height();

        qreal scale_factor;
        if(width_ratio <= height_ratio)
            scale_factor = width_ratio;
        else
            scale_factor = height_ratio;

        ui->graphicsView->scale(scale_factor, scale_factor);
    }

    scene->update();
    ui->graphicsView->show();
}

void SupervisionWindow::loadClassesToComboBox(QHash<int, QColor> hashLabelId2LabelColor, QHash<int, QString> hashLabelId2LabelName, int currentTrueLabel)
{
    qDebug();

    int index = 0;
    for (int j = 0; j < hashLabelId2LabelName.size(); j++) {
        ui->comboBoxClassForSupervision->addItem(hashLabelId2LabelName[j], hashLabelId2LabelColor[j]);
        const QModelIndex idx = ui->comboBoxClassForSupervision->model()->index(index++, 0);
        ui->comboBoxClassForSupervision->model()->setData(idx, hashLabelId2LabelColor[j], Qt::DecorationRole);
    }
    ui->comboBoxClassForSupervision->setCurrentIndex(currentTrueLabel);
}

void SupervisionWindow::on_pushButtonSupervise_clicked()
{
    qDebug();

    chosenClass = ui->comboBoxClassForSupervision->currentIndex();
    QDialog::accept();
}

void SupervisionWindow::on_pushButtonCancel_clicked()
{
    qDebug();

    QDialog::reject();
}

void SupervisionWindow::enableWidgetsByMode(QString mode)
{
    qDebug();

    if (mode == "2DImage"){
        ui->comboBoxPlane->setEnabled(false);
        ui->spinBoxSlice->setEnabled(false);
        ui->spinBoxBand->setEnabled(false);
    } else if (mode == "3DImage"){
        ui->comboBoxPlane->setEnabled(true);
        ui->spinBoxSlice->setEnabled(true);
        ui->spinBoxBand->setEnabled(false);
    } else if (mode == "MultBand2DImage"){
        ui->comboBoxPlane->setEnabled(false);
        ui->spinBoxSlice->setEnabled(false);
        ui->spinBoxBand->setEnabled(true);
    } else if (mode == "MultBand3DImage"){
        ui->comboBoxPlane->setEnabled(true);
        ui->spinBoxSlice->setEnabled(true);
        ui->spinBoxBand->setEnabled(true);
    }
}

void SupervisionWindow::on_pushButtonZoomIn_clicked()
{
    qDebug();
    qreal scaleFactor = 1.25;
    abs_scaleFactor *= scaleFactor;
    ui->graphicsView->scale(scaleFactor,scaleFactor);
}

void SupervisionWindow::on_pushButtonZoomOut_clicked()
{
    qDebug();
    qreal scaleFactor = 0.8;
    abs_scaleFactor *= scaleFactor;
    ui->graphicsView->scale(scaleFactor,scaleFactor);
}

void SupervisionWindow::on_pushButtonOriginalSize_clicked()
{
    qDebug();
    qreal scaleFactor = 1.0/abs_scaleFactor;
    ui->graphicsView->scale(scaleFactor,scaleFactor);
    abs_scaleFactor = 1.0;
}

void SupervisionWindow::on_pushButtonFitWindow_clicked()
{
    qDebug();
    qreal width_ratio = (float)ui->graphicsView->geometry().width() / ((float)image->width()*abs_scaleFactor);
    qreal height_ratio = (float)ui->graphicsView->geometry().height() / ((float)image->height()*abs_scaleFactor);

    printf("%f,%f\n",width_ratio,height_ratio);
    fflush(stdout);

    qreal scale_factor;
    if (width_ratio <= height_ratio)
        scale_factor = width_ratio;
    else
        scale_factor = height_ratio;

    abs_scaleFactor *= scale_factor;

    ui->graphicsView->scale(scale_factor, scale_factor);
}

void SupervisionWindow::update2DImage()
{
    if (image != nullptr)
        delete image;

    image = iftImageToQImage(img);

    scene->clear();
    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*image));
    scene->addItem(item);
    scene->update();
}

void SupervisionWindow::update3DImage()
{
    int index = ui->spinBoxSlice->value();

    iftImage *slice = nullptr;
    if (plane == "Axial"){
        if (index >= img->zsize){
            index = img->zsize-1;
            ui->spinBoxSlice->setValue(index);
        }
        slice = iftGetXYSlice(img,index);
    } else if (plane == "Coronal"){
        if (index >= img->ysize){
            index = img->ysize-1;
            ui->spinBoxSlice->setValue(index);
        }
        slice = iftGetZXSlice(img,index);
    }else if (plane == "Sagittal"){
        if (index >= img->xsize){
            index = img->xsize-1;
            ui->spinBoxSlice->setValue(index);
        }
        slice = iftGetYZSlice(img,index);
    }

    if (image != nullptr)
        delete image;

    image = iftImageToQImage(slice);

    scene->clear();
    QGraphicsPixmapItem* item = new QGraphicsPixmapItem(QPixmap::fromImage(*image));
    scene->addItem(item);
    scene->update();

    iftDestroyImage(&slice);
}

void SupervisionWindow::on_spinBoxSlice_valueChanged(int arg1)
{
    update3DImage();
}

void SupervisionWindow::on_comboBoxPlane_currentIndexChanged(const QString &arg1)
{
    if (arg1 == "Axial"){
        plane = "Axial";
        ui->spinBoxSlice->setMaximum(img->zsize-1);
    } else if (arg1 == "Coronal"){
        plane = "Coronal";
        ui->spinBoxSlice->setMaximum(img->ysize-1);
    } else {
        plane = "Sagittal";
        ui->spinBoxSlice->setMaximum(img->xsize-1);
    }

    update3DImage();
}

void SupervisionWindow::on_spinBoxBand_valueChanged(int arg1)
{
    if (img != nullptr)
        iftDestroyImage(&img);

    img = iftMImageToImage(mimg,255,arg1);

    if (iftIs3DImage(img)){
        update3DImage();
    } else {
        update2DImage();
    }
}
