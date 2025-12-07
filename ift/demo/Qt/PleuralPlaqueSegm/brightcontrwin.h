#ifndef BRIGHTCONTRWIN_H
#define BRIGHTCONTRWIN_H

#include <QDialog>

namespace Ui {
class BrightContrWin;
}

class BrightContrWin : public QDialog
{
    Q_OBJECT

public:
    explicit BrightContrWin(QWidget *parent = 0);
    ~BrightContrWin();
    int pastBrightValue;
    float pastContrastValue;

public slots:
    void callSignalBright();
    void callSignalContrast();
    void stepBrightness();
    void stepContrast();

signals:
    void changeBrightness(int value);
    void changeContrast(double value);

private:
    Ui::BrightContrWin *ui;
    void createActions();
};

#endif // BRIGHTCONTRWIN_H
