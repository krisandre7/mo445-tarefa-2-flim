/********************************************************************************
** Form generated from reading UI file 'brightcontrwin.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_BRIGHTCONTRWIN_H
#define UI_BRIGHTCONTRWIN_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QDialogButtonBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>

QT_BEGIN_NAMESPACE

class Ui_BrightContrWin
{
public:
    QDialogButtonBox *buttonBox;
    QLabel *lblBrightness;
    QSlider *hsBrightness;
    QLabel *lblContrast;
    QSlider *hsContrast;
    QSpinBox *sbBrightness;
    QSpinBox *sbContrast;

    void setupUi(QDialog *BrightContrWin)
    {
        if (BrightContrWin->objectName().isEmpty())
            BrightContrWin->setObjectName(QStringLiteral("BrightContrWin"));
        BrightContrWin->resize(400, 191);
        buttonBox = new QDialogButtonBox(BrightContrWin);
        buttonBox->setObjectName(QStringLiteral("buttonBox"));
        buttonBox->setGeometry(QRect(30, 150, 341, 32));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);
        lblBrightness = new QLabel(BrightContrWin);
        lblBrightness->setObjectName(QStringLiteral("lblBrightness"));
        lblBrightness->setGeometry(QRect(20, 20, 81, 17));
        hsBrightness = new QSlider(BrightContrWin);
        hsBrightness->setObjectName(QStringLiteral("hsBrightness"));
        hsBrightness->setGeometry(QRect(20, 40, 351, 29));
        hsBrightness->setMinimum(0);
        hsBrightness->setMaximum(100);
        hsBrightness->setValue(50);
        hsBrightness->setTracking(false);
        hsBrightness->setOrientation(Qt::Horizontal);
        lblContrast = new QLabel(BrightContrWin);
        lblContrast->setObjectName(QStringLiteral("lblContrast"));
        lblContrast->setGeometry(QRect(20, 80, 67, 17));
        hsContrast = new QSlider(BrightContrWin);
        hsContrast->setObjectName(QStringLiteral("hsContrast"));
        hsContrast->setGeometry(QRect(20, 100, 351, 29));
        hsContrast->setMaximum(100);
        hsContrast->setValue(0);
        hsContrast->setTracking(false);
        hsContrast->setOrientation(Qt::Horizontal);
        sbBrightness = new QSpinBox(BrightContrWin);
        sbBrightness->setObjectName(QStringLiteral("sbBrightness"));
        sbBrightness->setGeometry(QRect(120, 20, 48, 20));
        sbBrightness->setValue(50);
        sbContrast = new QSpinBox(BrightContrWin);
        sbContrast->setObjectName(QStringLiteral("sbContrast"));
        sbContrast->setGeometry(QRect(120, 80, 48, 20));
        sbContrast->setMaximum(100);
        sbContrast->setValue(0);

        retranslateUi(BrightContrWin);
        QObject::connect(buttonBox, SIGNAL(accepted()), BrightContrWin, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), BrightContrWin, SLOT(reject()));

        QMetaObject::connectSlotsByName(BrightContrWin);
    } // setupUi

    void retranslateUi(QDialog *BrightContrWin)
    {
        BrightContrWin->setWindowTitle(QApplication::translate("BrightContrWin", "Dialog", 0));
        lblBrightness->setText(QApplication::translate("BrightContrWin", "Brightness", 0));
        lblContrast->setText(QApplication::translate("BrightContrWin", "Contrast", 0));
    } // retranslateUi

};

namespace Ui {
    class BrightContrWin: public Ui_BrightContrWin {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_BRIGHTCONTRWIN_H
