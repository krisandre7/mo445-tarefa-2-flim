#include "brightcontrwin.h"
#include "ui_brightcontrwin.h"

BrightContrWin::BrightContrWin(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::BrightContrWin)
{
    ui->setupUi(this);
    pastBrightValue = 50;
    pastContrastValue = 1.0;
    createActions();
}

void BrightContrWin::callSignalBright()
{
    ui->sbBrightness->setValue(ui->hsBrightness->value());
    emit changeBrightness(ui->hsBrightness->value()-pastBrightValue);
    pastBrightValue = ui->hsBrightness->value();
}

void BrightContrWin::callSignalContrast()
{
    float newContrastValue = ((double)ui->hsContrast->value()/100)+1;
    ui->sbContrast->setValue(ui->hsContrast->value());
    emit changeContrast(newContrastValue/pastContrastValue);
    pastContrastValue = newContrastValue;
}

void BrightContrWin::stepBrightness()
{
    ui->hsBrightness->setValue(ui->sbBrightness->value());
}

void BrightContrWin::stepContrast()
{
    ui->hsContrast->setValue(ui->sbContrast->value());
}

void BrightContrWin::createActions()
{
    connect(ui->hsBrightness,SIGNAL(valueChanged(int)),this,SLOT(callSignalBright()));
    connect(ui->hsContrast,SIGNAL(valueChanged(int)),this,SLOT(callSignalContrast()));
    connect(ui->sbBrightness,SIGNAL(valueChanged(int)),this,SLOT(stepBrightness()));
    connect(ui->sbContrast,SIGNAL(valueChanged(int)),this,SLOT(stepContrast()));
}

BrightContrWin::~BrightContrWin()
{
    delete ui;
}
