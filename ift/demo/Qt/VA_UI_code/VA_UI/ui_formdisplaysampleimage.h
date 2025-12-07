/********************************************************************************
** Form generated from reading UI file 'formdisplaysampleimage.ui'
**
** Created by: Qt User Interface Compiler version 5.7.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_FORMDISPLAYSAMPLEIMAGE_H
#define UI_FORMDISPLAYSAMPLEIMAGE_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_FormDisplaySampleImage
{
public:
    QScrollArea *scrollAreaSampleImage;
    QWidget *scrollAreaWidgetContents;

    void setupUi(QWidget *FormDisplaySampleImage)
    {
        if (FormDisplaySampleImage->objectName().isEmpty())
            FormDisplaySampleImage->setObjectName(QStringLiteral("FormDisplaySampleImage"));
        FormDisplaySampleImage->resize(400, 300);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(FormDisplaySampleImage->sizePolicy().hasHeightForWidth());
        FormDisplaySampleImage->setSizePolicy(sizePolicy);
        scrollAreaSampleImage = new QScrollArea(FormDisplaySampleImage);
        scrollAreaSampleImage->setObjectName(QStringLiteral("scrollAreaSampleImage"));
        scrollAreaSampleImage->setGeometry(QRect(140, 130, 120, 80));
        scrollAreaSampleImage->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 118, 78));
        scrollAreaSampleImage->setWidget(scrollAreaWidgetContents);

        retranslateUi(FormDisplaySampleImage);

        QMetaObject::connectSlotsByName(FormDisplaySampleImage);
    } // setupUi

    void retranslateUi(QWidget *FormDisplaySampleImage)
    {
        FormDisplaySampleImage->setWindowTitle(QApplication::translate("FormDisplaySampleImage", "Form", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class FormDisplaySampleImage: public Ui_FormDisplaySampleImage {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_FORMDISPLAYSAMPLEIMAGE_H
