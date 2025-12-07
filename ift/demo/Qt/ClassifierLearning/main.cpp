#include "mainwindow.h"
#include <stdio.h>
#include <stdlib.h>
#include <QApplication>
#include <QtGlobal>
#include <QtCore>

FILE *filePtr = fopen("log_debug.txt", "w");

void myMessageOutput(QtMsgType type, const QMessageLogContext &context, const QString &msg)
{
    QString msg1 = msg.length() > 0 ? QString(" -> Msg: %1").arg(msg) : QString("");

    switch (type) {
    case QtDebugMsg:
        fprintf(filePtr, "Debug: %s:%u, %s%s\n", context.file, context.line, context.function, msg1.toLatin1().constData()) ; fflush(filePtr);
        break;
    case QtInfoMsg:
        fprintf(filePtr, "Info: %s:%u, %s%s\n", context.file, context.line, context.function, msg1.toLatin1().constData()) ; fflush(filePtr);
        break;
    case QtWarningMsg:
        fprintf(filePtr, "Warning: %s:%u, %s%s\n", context.file, context.line, context.function, msg1.toLatin1().constData()) ; fflush(filePtr);
        break;
    case QtCriticalMsg:
        fprintf(filePtr, "Critical: %s:%u, %s%s\n", context.file, context.line, context.function, msg1.toLatin1().constData()) ; fflush(filePtr);
        break;
    case QtFatalMsg:
        fprintf(filePtr, "Fatal: %s:%u, %s%s\n", context.file, context.line, context.function, msg1.toLatin1().constData()) ; fflush(filePtr);
        abort();
    }
}

int main(int argc, char *argv[])
{
    QApplication::setDesktopSettingsAware(false);
    QApplication a(argc, argv);
    qInstallMessageHandler(myMessageOutput);
    MainWindow w;
    w.show();

    return a.exec();
}
