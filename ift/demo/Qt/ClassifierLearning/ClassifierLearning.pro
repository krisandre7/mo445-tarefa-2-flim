#-------------------------------------------------
#
# Project created by QtCreator 2018-03-25T18:38:56
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ClassifierLearning
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

SOURCES += \
        main.cpp \
        mainwindow.cpp \
    graphnode.cpp \
    globalvariables.cpp \
    graphedge.cpp \
    customgraphicsview.cpp \
    supervisionwindow.cpp \
    safetyindicator.cpp

HEADERS += \
        mainwindow.h \
    graphnode.h \
    globalvariables.h \
    graphedge.h \
    customgraphicsview.h \
    supervisionwindow.h \
    safetyindicator.h

FORMS += \
        mainwindow.ui \
    supervisionwindow.ui

IFT_DIR = $$PWD/../../..

INCLUDEPATH += $$IFT_DIR/include

###INCLUDEPATH += /opt/local/include/       #macOS

INCLUDEPATH += $$IFT_DIR/externals/libsvm/include
INCLUDEPATH += $$IFT_DIR/externals/libpng/include
INCLUDEPATH += $$IFT_DIR/externals/libjpeg/include
INCLUDEPATH += $$IFT_DIR/externals/tsne/CPU/include
INCLUDEPATH += $$IFT_DIR/externals/tsne/GPU/src
INCLUDEPATH += $$IFT_DIR/externals/zlib/include
LIBS += -L $$IFT_DIR/lib -lift -llapack -lblas -lm -fopenmp -std=gnu++11

###QMAKE_CC = /opt/local/bin/gcc-mp-5       #macOS
###QMAKE_CXX = /opt/local/bin/g++-mp-5      #macOS
###QMAKE_CXXFLAGS_RELEASE = -pipe -stdlib=libstdc++ -g -std=gnu++11 $(EXPORT_ARCH_ARGS) -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -mmacosx-version-min=10.10 -Wall -W -fPIC $(DEFINES)         #macOS
QMAKE_CXXFLAGS = -fopenmp

CONFIG += c++-11

#GPU STUFFS
INCLUDEPATH += /usr/local/cuda/include
LIBS   += -L/usr/local/cuda/lib64 -lcublas -lcudart

RESOURCES += \
    myicons.qrc

DISTFILES += \
    Datasets/convnet_peixinho_perc0.05.zip
