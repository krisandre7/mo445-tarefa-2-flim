#ifndef VISVA_H
#define VISVA_H

#include <QMainWindow>
#include "global.h"
#include "brightcontrwin.h"

QT_BEGIN_NAMESPACE
class QScrollBar;
QT_END_NAMESPACE

namespace Ui {
class Visva;
}

class Visva : public QMainWindow
{
    Q_OBJECT

public:
    explicit Visva(QWidget *parent = 0);
    bool LoadFile(const QString &);
    ~Visva();

private slots:
    void OpenFile();
    void SaveFile();
    void NewFile();
    void ZoomIn();
    void ZoomOut();
    void NormalSize();
    void About();
    void OpenBrightContrWin();
    void Brightness(int value);
    void Contrast(double value);

private:
    double scaleFactor;
    void scaleImage(double factor);
    void createMenus();
    void createActions();

    Ui::Visva *ui;

    QImage imgLoad;
    QImage imgScaled;
    QImage imgAritmetic;

    BrightContrWin *brightcontrwin;

    QAction *openFileAct;
    QAction *saveFileAct;
    QAction *newFileAct;
    QAction *exitAct;
    QAction *zoomInAct;
    QAction *zoomOutAct;
    QAction *normalSizeAct;
    QAction *aboutAct;

    QMenu *fileMenu;
    QMenu *viewMenu;
    QMenu *helpMenu;
};

#endif // VISVA_H
