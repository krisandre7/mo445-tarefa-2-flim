#-------------------------------------------------
#
# Project created by QtCreator 2017-07-21T13:51:42
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Teste
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    mywidgetitem.cpp \
    mylistwidget.cpp \
    mygraphicwidget.cpp \
    mynode.cpp \
    qcustomplot.cpp \
    myedge.cpp \
    dialogsceneoptions.cpp \
    global.cpp

HEADERS += \
        mainwindow.h \
    mywidgetitem.h \
    mylistwidget.h \
    mygraphicwidget.h \
    mynode.h \
    qcustomplot.h \
    myedge.h \
    dialogsceneoptions.h \
    global.h

FORMS += \
        mainwindow.ui \
    dialogsceneoptions.ui

RESOURCES += \
    resource_images.qrc \
    datasets_folder.qrc

IFT_DIR = $$PWD/../../..

INCLUDEPATH += $$IFT_DIR/include

INCLUDEPATH += $$IFT_DIR/externals/libsvm/include
INCLUDEPATH += $$IFT_DIR/externals/libpng/include
INCLUDEPATH += $$IFT_DIR/externals/libjpeg/include
INCLUDEPATH += $$IFT_DIR/externals/tsne/CPU/include
INCLUDEPATH += $$IFT_DIR/externals/tsne/GPU/src
INCLUDEPATH += $$IFT_DIR/externals/zlib/include
LIBS += -L $$IFT_DIR/lib -lift -llapack -lblas -lm -fopenmp

#GPU STUFFS
INCLUDEPATH += /usr/local/cuda/include
LIBS   += -L/usr/local/cuda/lib64 -lcublas -lcudart

