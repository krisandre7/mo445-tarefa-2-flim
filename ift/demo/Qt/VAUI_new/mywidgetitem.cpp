#include "mywidgetitem.h"

MyWidgetItem::MyWidgetItem(QWidget *parent) : QWidget(parent)
{

}


MyWidgetItem::MyWidgetItem(QPixmap image, QVector<QString> labelNames,
                           QVector<QColor> labeColors, QWidget *parent) : QWidget(parent){

    QComboBox* comboBox = new QComboBox(this);
    int index = 0;
    for (int i = 0; i < labelNames.size(); ++i) {
        comboBox->addItem(labelNames.at(i),labeColors.at(i));
        const QModelIndex idx = comboBox->model()->index(index++, 0);
        comboBox->model()->setData(idx,labeColors.at(i), Qt::DecorationRole);
    }
    hasRepresentativeImage = false;
    QFont font;
    font.setPointSize(font.pointSize() - 2);
    comboBox->setFont(font);
    this->myComboBox = comboBox;
    this->labelImage = new QLabel();
    labelImage->setPixmap(image.scaled(itemIconSize_width,
                                       itemIconSize_height,
                                       Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation));
    setLayout(new QVBoxLayout(this));
    layout()->addWidget(labelImage);
    layout()->addWidget(myComboBox);
}

MyWidgetItem::MyWidgetItem(QString path, QVector<QString> labelNames,
                           QVector<QColor> labeColors, QWidget *parent) : QWidget(parent){
    path2Image = path;
    QPixmap image(path);
    QComboBox* comboBox = new QComboBox(this);
    hasRepresentativeImage = false;
    int index = 0;
    for (int i = 0; i < labelNames.size(); ++i) {
        comboBox->addItem(labelNames.at(i),labeColors.at(i));
        const QModelIndex idx = comboBox->model()->index(index++, 0);
        comboBox->model()->setData(idx,labeColors.at(i), Qt::DecorationRole);
    }
    QFont font;
    font.setPointSize(font.pointSize() - 2);
    comboBox->setFont(font);
    this->myComboBox = comboBox;
    this->labelImage = new QLabel();
    //labelImage->setToolTip(path2Image);
    labelImage->setPixmap(image.scaled(itemIconSize_width,
                                       itemIconSize_height,
                                       Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation));
    setLayout(new QVBoxLayout(this));
    layout()->addWidget(labelImage);
    layout()->addWidget(myComboBox);
}

void MyWidgetItem::updateImage(QString path){
    path2Image = path;
    QPixmap image(path);
    //labelImage->setToolTip(path2Image);
    labelImage->setPixmap(image.scaled(itemIconSize_width,
                                       itemIconSize_height,
                                       Qt::KeepAspectRatio,
                                       Qt::SmoothTransformation));
}

void MyWidgetItem::mousePressEvent(QMouseEvent *event){
    QWidget::mousePressEvent(event);
    //isSelected = true;
    //emit pressed();
}
