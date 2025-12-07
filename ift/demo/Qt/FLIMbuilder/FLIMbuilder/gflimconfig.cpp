#include "gflimconfig.h"
#include "ui_gflimconfig.h"

gflimconfig::gflimconfig(QWidget *parent, int spx, int initSeeds) :
    QDialog(parent),
    ui(new Ui::gflimconfig)
{
    ui->setupUi(this);

    ui->sb_initialSeedsDISF->setMaximum(100000);
    ui->sb_superpixel_number->setMaximum(99999);
    ui->sb_initialSeedsDISF->setMinimum(2);
    ui->sb_superpixel_number->setMinimum(2);
    ui->sb_initialSeedsDISF->setValue(initSeeds);
    ui->sb_superpixel_number->setValue(spx);
}

gflimconfig::~gflimconfig()
{
    delete ui;
}

int gflimconfig::getInitialSeedsDISF(){
    return ui->sb_initialSeedsDISF->value();
}

int gflimconfig::getSuperpixel_number(){
    return ui->sb_superpixel_number->value();
}

void gflimconfig::disable_sb_initialSeedsDISF(){
    ui->sb_initialSeedsDISF->setDisabled(true);
}

void gflimconfig::enable_sb_initialSeedsDISF(){
    ui->sb_initialSeedsDISF->setDisabled(false);
}

void gflimconfig::disable_sb_superpixel_number(){
    ui->sb_superpixel_number->setDisabled(true);
}

void gflimconfig::enable_sb_superpixel_number(){
    ui->sb_superpixel_number->setDisabled(false);
}
