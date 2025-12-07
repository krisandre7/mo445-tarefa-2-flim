#ifndef GFLIMCONFIG_H
#define GFLIMCONFIG_H

#include <QDialog>
#include "gflim.h"

namespace Ui {
class gflimconfig;
}

class gflimconfig : public QDialog
{
    Q_OBJECT

public:
    explicit gflimconfig(QWidget *parent = nullptr,
                         int spx = 1000,
                         int initSeeds = 1000);
    ~gflimconfig();

    int getInitialSeedsDISF();
    int getSuperpixel_number();
    void disable_sb_initialSeedsDISF();
    void enable_sb_initialSeedsDISF();
    void disable_sb_superpixel_number();
    void enable_sb_superpixel_number();

//private:
    Ui::gflimconfig *ui;
};

#endif // GFLIMCONFIG_H
