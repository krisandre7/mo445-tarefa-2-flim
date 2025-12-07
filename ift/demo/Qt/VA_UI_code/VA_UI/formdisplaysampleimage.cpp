#include "formdisplaysampleimage.h"
#include "ui_formdisplaysampleimage.h"

FormDisplaySampleImage::FormDisplaySampleImage(QWidget *parent,QString filePath) :
    QDialog(parent),
    ui(new Ui::FormDisplaySampleImage)
{
    ui->setupUi(this);
    QImage sampleImage(filePath.toLatin1().data());
    imageLabel = new QLabel("");
    imageLabel->setPixmap(QPixmap::fromImage(sampleImage));
    imageLabel->setGeometry(0,0,sampleImage.size().width(),sampleImage.size().height());
    imageLabel->setStyleSheet("border: none");
    areaMaximumHeight = 600;
    areaMaximumWidth = 800;
    scrollAreaHeight = sampleImage.size().height()+5;
    scrollAreaWidth = sampleImage.size().width()+5;
    windowHeight = sampleImage.size().height()+30;
    windowWidth = sampleImage.size().width()+30;

    if(areaMaximumHeight < scrollAreaHeight){
        scrollAreaHeight = areaMaximumHeight+5;
        windowHeight = areaMaximumHeight+30;
    }
    if(areaMaximumWidth < scrollAreaWidth){
        scrollAreaWidth = areaMaximumWidth+5;
        windowWidth = areaMaximumWidth+30;

    }
    FormDisplaySampleImage::setGeometry(0,0,windowWidth,windowHeight);
    ui->scrollAreaSampleImage->setGeometry(10,10,scrollAreaWidth,scrollAreaHeight);
    ui->scrollAreaSampleImage->setWidget(imageLabel);
    FormDisplaySampleImage::setWindowTitle("sample image");

}


FormDisplaySampleImage::~FormDisplaySampleImage()
{
    delete ui;
}
