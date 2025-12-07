#-------------------------------------------------
#
# Project created by QtCreator 2016-01-05T16:09:16
#
# Author: Azael de Melo e Sousa
#
# Lung Icon: Chris Dawson (Noun Project)
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = QtVisva
TEMPLATE = app

SOURCES += main.cpp\
        visva.cpp\
    global.cpp \
    brightcontrwin.cpp

HEADERS  += visva.h\
    global.h \
    brightcontrwin.h

FORMS    += visva.ui \
    brightcontrwin.ui

LIBS += -L$$PWD/../../../lib/ -lift -llapack -lblas -lm -fopenmp

INCLUDEPATH += $$PWD/../../../include

QMAKE_CFLAGS += -O3 -fopenmp -pthread -pedantic -std=c++11

QMAKE_CXXFLAGS += -O3 -fopenmp -pthread -pedantic -std=c++11
