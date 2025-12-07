/********************************************************************************
** Form generated from reading UI file 'dialogupdatesamplesimagedirectory.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOGUPDATESAMPLESIMAGEDIRECTORY_H
#define UI_DIALOGUPDATESAMPLESIMAGEDIRECTORY_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>

QT_BEGIN_NAMESPACE

class Ui_DialogUpdateSamplesImageDirectory
{
public:
    QLineEdit *lineEditPrefix;
    QLabel *label_2;
    QLabel *label_3;
    QLineEdit *lineEditDirectory;
    QPushButton *pushButtonBrowserDirectory;
    QPushButton *pushButtonCancel;
    QPushButton *pushButtonOk;

    void setupUi(QDialog *DialogUpdateSamplesImageDirectory)
    {
        if (DialogUpdateSamplesImageDirectory->objectName().isEmpty())
            DialogUpdateSamplesImageDirectory->setObjectName(QStringLiteral("DialogUpdateSamplesImageDirectory"));
        DialogUpdateSamplesImageDirectory->resize(299, 199);
        lineEditPrefix = new QLineEdit(DialogUpdateSamplesImageDirectory);
        lineEditPrefix->setObjectName(QStringLiteral("lineEditPrefix"));
        lineEditPrefix->setGeometry(QRect(10, 30, 200, 24));
        lineEditPrefix->setContextMenuPolicy(Qt::DefaultContextMenu);
        label_2 = new QLabel(DialogUpdateSamplesImageDirectory);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(10, 10, 59, 14));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        label_2->setFont(font);
        label_3 = new QLabel(DialogUpdateSamplesImageDirectory);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(10, 80, 121, 16));
        label_3->setFont(font);
        lineEditDirectory = new QLineEdit(DialogUpdateSamplesImageDirectory);
        lineEditDirectory->setObjectName(QStringLiteral("lineEditDirectory"));
        lineEditDirectory->setGeometry(QRect(10, 100, 200, 24));
        pushButtonBrowserDirectory = new QPushButton(DialogUpdateSamplesImageDirectory);
        pushButtonBrowserDirectory->setObjectName(QStringLiteral("pushButtonBrowserDirectory"));
        pushButtonBrowserDirectory->setGeometry(QRect(220, 100, 70, 24));
        QFont font1;
        font1.setBold(true);
        font1.setItalic(false);
        font1.setUnderline(false);
        font1.setWeight(75);
        font1.setStrikeOut(false);
        font1.setKerning(true);
        pushButtonBrowserDirectory->setFont(font1);
        pushButtonCancel = new QPushButton(DialogUpdateSamplesImageDirectory);
        pushButtonCancel->setObjectName(QStringLiteral("pushButtonCancel"));
        pushButtonCancel->setGeometry(QRect(110, 170, 80, 22));
        pushButtonOk = new QPushButton(DialogUpdateSamplesImageDirectory);
        pushButtonOk->setObjectName(QStringLiteral("pushButtonOk"));
        pushButtonOk->setGeometry(QRect(200, 170, 80, 22));

        retranslateUi(DialogUpdateSamplesImageDirectory);

        QMetaObject::connectSlotsByName(DialogUpdateSamplesImageDirectory);
    } // setupUi

    void retranslateUi(QDialog *DialogUpdateSamplesImageDirectory)
    {
        DialogUpdateSamplesImageDirectory->setWindowTitle(QApplication::translate("DialogUpdateSamplesImageDirectory", "Dialog", Q_NULLPTR));
        lineEditPrefix->setText(QString());
        label_2->setText(QApplication::translate("DialogUpdateSamplesImageDirectory", "Prefix", Q_NULLPTR));
        label_3->setText(QApplication::translate("DialogUpdateSamplesImageDirectory", "Images Directory", Q_NULLPTR));
        pushButtonBrowserDirectory->setText(QApplication::translate("DialogUpdateSamplesImageDirectory", "Browser", Q_NULLPTR));
        pushButtonCancel->setText(QApplication::translate("DialogUpdateSamplesImageDirectory", "Cancel", Q_NULLPTR));
        pushButtonOk->setText(QApplication::translate("DialogUpdateSamplesImageDirectory", "Ok", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class DialogUpdateSamplesImageDirectory: public Ui_DialogUpdateSamplesImageDirectory {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOGUPDATESAMPLESIMAGEDIRECTORY_H
