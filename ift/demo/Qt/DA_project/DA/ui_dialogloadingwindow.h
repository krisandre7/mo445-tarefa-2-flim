/********************************************************************************
** Form generated from reading UI file 'dialogloadingwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOGLOADINGWINDOW_H
#define UI_DIALOGLOADINGWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QProgressBar>

QT_BEGIN_NAMESPACE

class Ui_DialogLoadingWindow
{
public:
    QProgressBar *progressBar;

    void setupUi(QDialog *DialogLoadingWindow)
    {
        if (DialogLoadingWindow->objectName().isEmpty())
            DialogLoadingWindow->setObjectName(QStringLiteral("DialogLoadingWindow"));
        DialogLoadingWindow->resize(318, 75);
        progressBar = new QProgressBar(DialogLoadingWindow);
        progressBar->setObjectName(QStringLiteral("progressBar"));
        progressBar->setGeometry(QRect(20, 20, 281, 23));
        progressBar->setMaximum(10);
        progressBar->setValue(1);

        retranslateUi(DialogLoadingWindow);

        QMetaObject::connectSlotsByName(DialogLoadingWindow);
    } // setupUi

    void retranslateUi(QDialog *DialogLoadingWindow)
    {
        DialogLoadingWindow->setWindowTitle(QApplication::translate("DialogLoadingWindow", "Dialog", Q_NULLPTR));
        progressBar->setFormat(QApplication::translate("DialogLoadingWindow", "%v/%m", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DialogLoadingWindow: public Ui_DialogLoadingWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOGLOADINGWINDOW_H
