#-------------------------------------------------
#
# Project created by QtCreator 2019-06-02T10:09:44
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = skel-viz
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11


SOURCES += \
        main.cpp \
        mainwindow.cpp \
        iftqtapi.cpp

HEADERS += \
        mainwindow.h \
        iftqtapi.h

FORMS += \
        mainwindow.ui

IFT_DIR = $$PWD/../../../..

LIBS += -L$$IFT_DIR/lib/ -lift -llapack -lblas -lcblas -lm -fopenmp

INCLUDEPATH += $$IFT_DIR/include
INCLUDEPATH += $$IFT_DIR/externals/libsvm/include
INCLUDEPATH += $$IFT_DIR/externals/libpng/include
INCLUDEPATH += $$IFT_DIR/externals/libjpeg/include
INCLUDEPATH += $$IFT_DIR/externals/zlib/include
INCLUDEPATH += $$IFT_DIR/externals/tsne/include

QMAKE_CFLAGS += -fopenmp -pthread -pedantic -std=c++11
QMAKE_CXXFLAGS += -fopenmp -pthread -pedantic -std=c++11

mac { 
        QMAKE_CC = gcc-9
        QMAKE_CXX = g++-9
        QMAKE_LINK = g++-9
        QMAKE_CXXFLAGS -= -stdlib=libc++
        QMAKE_LFLAGS -= -stdlib=libc++
        INCLUDEPATH += /usr/local/opt/openblas/include/
        LIBS += -lomp
        QMAKE_CFLAGS += -lomp
        QMAKE_MACOSX_DEPLOYMENT_TARGET = 10.14
}

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
