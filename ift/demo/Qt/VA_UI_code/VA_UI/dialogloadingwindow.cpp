#include "dialogloadingwindow.h"
#include "ui_dialogloadingwindow.h"

DialogLoadingWindow::DialogLoadingWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogLoadingWindow)
{
    ui->setupUi(this);
}

DialogLoadingWindow::~DialogLoadingWindow()
{
    delete ui;
}

void DialogLoadingWindow::updateProgressBar(int percentage){
    ui->progressBar->setValue(percentage);
}
