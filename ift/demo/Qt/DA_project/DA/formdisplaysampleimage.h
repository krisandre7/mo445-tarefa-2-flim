#ifndef FORMDISPLAYSAMPLEIMAGE_H
#define FORMDISPLAYSAMPLEIMAGE_H

#include <QWidget>
#include <QDialog>
#include <QScrollArea>
#include <QLabel>
#include <QGridLayout>


namespace Ui {
class FormDisplaySampleImage;
}

class FormDisplaySampleImage : public QDialog
{

public:
    FormDisplaySampleImage(QWidget *parent = 0, QString filePath=" ");
    ~FormDisplaySampleImage();
    QLabel *imageLabel;

private:
    Ui::FormDisplaySampleImage *ui;
    int areaMaximumHeight;
    int areaMaximumWidth;
    int scrollAreaWidth;
    int scrollAreaHeight;
    int windowWidth;
    int windowHeight;

};

#endif // FORMDISPLAYSAMPLEIMAGE_H
