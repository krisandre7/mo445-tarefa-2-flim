#include "dialogupdatesamplesimagedirectory.h"
#include "ui_dialogupdatesamplesimagedirectory.h"

DialogUpdateSamplesImageDirectory::DialogUpdateSamplesImageDirectory(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DialogUpdateSamplesImageDirectory)
{
    ui->setupUi(this);
}

DialogUpdateSamplesImageDirectory::DialogUpdateSamplesImageDirectory(QWidget *parent, QString prefix,QString directory) :
    QDialog(parent),
    ui(new Ui::DialogUpdateSamplesImageDirectory)
{
    ui->setupUi(this);
    ui->lineEditPrefix->setText(prefix.toLatin1().data());
    ui->lineEditDirectory->setText(directory.toLatin1().data());
}

DialogUpdateSamplesImageDirectory::~DialogUpdateSamplesImageDirectory()
{
    delete ui;
}



void DialogUpdateSamplesImageDirectory::on_pushButtonBrowserDirectory_clicked()
{
    QString directoryName = QFileDialog::getExistingDirectory(this, tr("Open Directory"),
                                                              "/home",
                                                              QFileDialog::ShowDirsOnly
                                                              | QFileDialog::DontResolveSymlinks);
    ui->lineEditDirectory->setText(directoryName);
}

void DialogUpdateSamplesImageDirectory::on_pushButtonCancel_clicked()
{
    //emit windowClosed();
    DialogUpdateSamplesImageDirectory::close();
}

void DialogUpdateSamplesImageDirectory::on_pushButtonOk_clicked()
{
    emit updateDirectoryAndPrefix(ui->lineEditPrefix->text(),ui->lineEditDirectory->text());
    DialogUpdateSamplesImageDirectory::close();
}

void DialogUpdateSamplesImageDirectory::closeEvent(QCloseEvent *event){
    QDialog::closeEvent(event);
    emit windowClosed();
}
