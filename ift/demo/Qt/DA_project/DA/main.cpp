#include "mainwindow.h"
#include <QApplication>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    //w.installEventFilter(&resizeFilter);
    w.showMaximized();
    //w.show();
    //fprintf(stderr,"%d %d\n", w.height(),w.width());
    return a.exec();
}
