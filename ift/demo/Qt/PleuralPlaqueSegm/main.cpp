#include "visva.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    Visva w;
    w.show();

    return a.exec();
}
