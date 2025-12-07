#-------------------------------------------------
#
# Project created by QtCreator 2017-01-26T23:45:47
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = imageViewer
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    global.cpp \
    clickablelabel.cpp

HEADERS  += mainwindow.h \
    global.h \
    clickablelabel.h

FORMS    += mainwindow.ui

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../lib/release/ -lift
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../lib/debug/ -lift
else:unix: LIBS += -L$$PWD/../../../../lib/ -lift

INCLUDEPATH += $$PWD/../../../../include
DEPENDPATH += $$PWD/../../../../include

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/release/libift.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/debug/libift.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/release/ift.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/debug/ift.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../lib/libift.a

unix|win32: LIBS += -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lpng -lz -lm -fopenmp

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../libsvm/release/ -lsvm
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../libsvm/debug/ -lsvm
else:unix: LIBS += -L$$PWD/../../../../libsvm/ -lsvm

INCLUDEPATH += $$PWD/../../../../libsvm
DEPENDPATH += $$PWD/../../../../libsvm

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../libsvm/release/libsvm.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../libsvm/debug/libsvm.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../libsvm/release/svm.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../libsvm/debug/svm.lib
else:unix: PRE_TARGETDEPS += $$PWD/../../../../libsvm/libsvm.a
