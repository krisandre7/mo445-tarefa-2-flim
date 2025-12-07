#-------------------------------------------------
#
# Project created by QtCreator 2017-01-19T15:33:14
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = VA_UI
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    global.cpp \
    resizefilter.cpp \
    formdisplaysampleimage.cpp \
    dialogupdatesamplesimagedirectory.cpp \
    dialogloadingwindow.cpp \
    mygraphicscene.cpp \
    graphicsviewzoom.cpp \
    edge.cpp \
    node.cpp \
    graphwidget.cpp \
    activelearningutils.cpp \
    colorlisteditor.cpp \
    qcustomplot.cpp \
    sceneactionmenu.cpp


HEADERS  += mainwindow.h \
    global.h \
    resizefilter.h \
    formdisplaysampleimage.h \
    dialogupdatesamplesimagedirectory.h \
    dialogloadingwindow.h \
    mygraphicscene.h \
    graphicsviewzoom.h \
    edge.h \
    node.h \
    graphwidget.h \
    activelearningutils.h \
    colorlisteditor.h \
    qcustomplot.h \
    sceneactionmenu.h


FORMS    += mainwindow.ui \
    formdisplaysampleimage.ui \
    dialogupdatesamplesimagedirectory.ui \
    dialogloadingwindow.ui


IFT_DIR = $$PWD/../../../..


#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../lib/release/ -lift
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../lib/debug/ -lift
#else:unix: LIBS += -L$$PWD/../../../../lib/ -lift

#win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/release/libift.a
#else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/debug/libift.a
#else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/release/ift.lib
#else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../../../../lib/debug/ift.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../../../../lib/libift.a


#unix|win32: LIBS += -L$(NEWIFT_DIR)/lib -lift -llapack -lblas -lm -fopenmp

#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../libsvm/release/ -lsvm
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../libsvm/debug/ -lsvm
#else:unix: LIBS += -L$$PWD/../../../../libsvm/ -lsvm



INCLUDEPATH += $$IFT_DIR/include
LIBPATH  += $$IFT_DIR/lib

INCLUDEPATH += $$IFT_DIR/externals/libsvm/include
INCLUDEPATH += $$IFT_DIR/externals/libpng/include
INCLUDEPATH += $$IFT_DIR/externals/libjpeg/include
INCLUDEPATH += $$IFT_DIR/externals/tsne/include
INCLUDEPATH += $$IFT_DIR/externals/zlib/include

#LIBPATH += $$IFT_DIR/externals/libsvm/lib
#LIBPATH += $$IFT_DIR/externals/libpng/lib
#LIBPATH += $$IFT_DIR/externals/libjpeg/lib
#LIBPATH += $$IFT_DIR/externals/zlib/lib

LIBS += -lift -llapack -lblas -lm -fopenmp
