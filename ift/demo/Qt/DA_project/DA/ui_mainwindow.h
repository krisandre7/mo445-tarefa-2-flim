/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "datasetviewer.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionDataSet;
    QAction *actionDataset_Images_Directory_2;
    QAction *actionCategories_2;
    QAction *actionSave_dataset;
    QAction *actionDataset_Test;
    QAction *actionDataset_Validation;
    QWidget *centralWidget;
    QHBoxLayout *horizontalLayout_2;
    QHBoxLayout *horizontalLayout;
    DataSetViewer *graphicsViewProjectionArea;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout_3;
    QFormLayout *formLayout_2;
    QLabel *label;
    QLineEdit *lineEdit;
    QLabel *label_2;
    QLineEdit *lineEdit_2;
    QPushButton *pushButton_4;
    QVBoxLayout *verticalLayout_4;
    QPushButton *pushButton_3;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_2;
    QFormLayout *formLayout;
    QLabel *kNearestLabel;
    QSpinBox *kNearestSpinBox;
    QLabel *preProcessLabel;
    QComboBox *preProcessComboBox;
    QLabel *pca_componentsLabel;
    QDoubleSpinBox *pca_componentsDoubleSpinBox;
    QMenuBar *menuBar;
    QMenu *menuOpen;
    QMenu *menuOpen_2;
    QMenu *menuSave;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(1316, 715);
        MainWindow->setAutoFillBackground(false);
        actionDataSet = new QAction(MainWindow);
        actionDataSet->setObjectName(QStringLiteral("actionDataSet"));
        actionDataset_Images_Directory_2 = new QAction(MainWindow);
        actionDataset_Images_Directory_2->setObjectName(QStringLiteral("actionDataset_Images_Directory_2"));
        actionCategories_2 = new QAction(MainWindow);
        actionCategories_2->setObjectName(QStringLiteral("actionCategories_2"));
        actionSave_dataset = new QAction(MainWindow);
        actionSave_dataset->setObjectName(QStringLiteral("actionSave_dataset"));
        actionDataset_Test = new QAction(MainWindow);
        actionDataset_Test->setObjectName(QStringLiteral("actionDataset_Test"));
        actionDataset_Validation = new QAction(MainWindow);
        actionDataset_Validation->setObjectName(QStringLiteral("actionDataset_Validation"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        horizontalLayout_2 = new QHBoxLayout(centralWidget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        graphicsViewProjectionArea = new DataSetViewer(centralWidget);
        graphicsViewProjectionArea->setObjectName(QStringLiteral("graphicsViewProjectionArea"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(1);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(graphicsViewProjectionArea->sizePolicy().hasHeightForWidth());
        graphicsViewProjectionArea->setSizePolicy(sizePolicy1);
        graphicsViewProjectionArea->setContextMenuPolicy(Qt::CustomContextMenu);
        graphicsViewProjectionArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        graphicsViewProjectionArea->setDragMode(QGraphicsView::RubberBandDrag);
        graphicsViewProjectionArea->setCacheMode(QGraphicsView::CacheBackground);
        graphicsViewProjectionArea->setTransformationAnchor(QGraphicsView::NoAnchor);
        graphicsViewProjectionArea->setResizeAnchor(QGraphicsView::AnchorViewCenter);
        graphicsViewProjectionArea->setViewportUpdateMode(QGraphicsView::MinimalViewportUpdate);
        graphicsViewProjectionArea->setRubberBandSelectionMode(Qt::IntersectsItemShape);

        horizontalLayout->addWidget(graphicsViewProjectionArea);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        QSizePolicy sizePolicy2(QSizePolicy::Minimum, QSizePolicy::Minimum);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy2);
        verticalLayout_3 = new QVBoxLayout(groupBox_2);
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setContentsMargins(11, 11, 11, 11);
        verticalLayout_3->setObjectName(QStringLiteral("verticalLayout_3"));
        formLayout_2 = new QFormLayout();
        formLayout_2->setSpacing(6);
        formLayout_2->setObjectName(QStringLiteral("formLayout_2"));
        label = new QLabel(groupBox_2);
        label->setObjectName(QStringLiteral("label"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, label);

        lineEdit = new QLineEdit(groupBox_2);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(lineEdit->sizePolicy().hasHeightForWidth());
        lineEdit->setSizePolicy(sizePolicy3);

        formLayout_2->setWidget(0, QFormLayout::FieldRole, lineEdit);

        label_2 = new QLabel(groupBox_2);
        label_2->setObjectName(QStringLiteral("label_2"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, label_2);

        lineEdit_2 = new QLineEdit(groupBox_2);
        lineEdit_2->setObjectName(QStringLiteral("lineEdit_2"));

        formLayout_2->setWidget(1, QFormLayout::FieldRole, lineEdit_2);

        pushButton_4 = new QPushButton(groupBox_2);
        pushButton_4->setObjectName(QStringLiteral("pushButton_4"));

        formLayout_2->setWidget(2, QFormLayout::FieldRole, pushButton_4);


        verticalLayout_3->addLayout(formLayout_2);

        verticalLayout_4 = new QVBoxLayout();
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setObjectName(QStringLiteral("verticalLayout_4"));
        pushButton_3 = new QPushButton(groupBox_2);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));

        verticalLayout_4->addWidget(pushButton_3);

        pushButton = new QPushButton(groupBox_2);
        pushButton->setObjectName(QStringLiteral("pushButton"));

        verticalLayout_4->addWidget(pushButton);

        pushButton_2 = new QPushButton(groupBox_2);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));

        verticalLayout_4->addWidget(pushButton_2);


        verticalLayout_3->addLayout(verticalLayout_4);


        verticalLayout->addWidget(groupBox_2);

        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        sizePolicy2.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy2);
        verticalLayout_2 = new QVBoxLayout(groupBox);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QStringLiteral("verticalLayout_2"));
        formLayout = new QFormLayout();
        formLayout->setSpacing(6);
        formLayout->setObjectName(QStringLiteral("formLayout"));
        kNearestLabel = new QLabel(groupBox);
        kNearestLabel->setObjectName(QStringLiteral("kNearestLabel"));

        formLayout->setWidget(0, QFormLayout::LabelRole, kNearestLabel);

        kNearestSpinBox = new QSpinBox(groupBox);
        kNearestSpinBox->setObjectName(QStringLiteral("kNearestSpinBox"));
        kNearestSpinBox->setMinimum(1);
        kNearestSpinBox->setMaximum(99999999);
        kNearestSpinBox->setValue(1);

        formLayout->setWidget(0, QFormLayout::FieldRole, kNearestSpinBox);

        preProcessLabel = new QLabel(groupBox);
        preProcessLabel->setObjectName(QStringLiteral("preProcessLabel"));

        formLayout->setWidget(1, QFormLayout::LabelRole, preProcessLabel);

        preProcessComboBox = new QComboBox(groupBox);
        preProcessComboBox->setObjectName(QStringLiteral("preProcessComboBox"));

        formLayout->setWidget(1, QFormLayout::FieldRole, preProcessComboBox);

        pca_componentsLabel = new QLabel(groupBox);
        pca_componentsLabel->setObjectName(QStringLiteral("pca_componentsLabel"));

        formLayout->setWidget(2, QFormLayout::LabelRole, pca_componentsLabel);

        pca_componentsDoubleSpinBox = new QDoubleSpinBox(groupBox);
        pca_componentsDoubleSpinBox->setObjectName(QStringLiteral("pca_componentsDoubleSpinBox"));
        pca_componentsDoubleSpinBox->setDecimals(3);
        pca_componentsDoubleSpinBox->setMaximum(1);
        pca_componentsDoubleSpinBox->setSingleStep(0.1);
        pca_componentsDoubleSpinBox->setValue(0.99);

        formLayout->setWidget(2, QFormLayout::FieldRole, pca_componentsDoubleSpinBox);


        verticalLayout_2->addLayout(formLayout);


        verticalLayout->addWidget(groupBox);


        horizontalLayout->addLayout(verticalLayout);


        horizontalLayout_2->addLayout(horizontalLayout);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1316, 25));
        menuOpen = new QMenu(menuBar);
        menuOpen->setObjectName(QStringLiteral("menuOpen"));
        menuOpen_2 = new QMenu(menuOpen);
        menuOpen_2->setObjectName(QStringLiteral("menuOpen_2"));
        menuSave = new QMenu(menuOpen);
        menuSave->setObjectName(QStringLiteral("menuSave"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuOpen->menuAction());
        menuOpen->addAction(menuOpen_2->menuAction());
        menuOpen->addAction(menuSave->menuAction());
        menuOpen_2->addAction(actionDataSet);
        menuOpen_2->addAction(actionDataset_Images_Directory_2);
        menuOpen_2->addAction(actionCategories_2);
        menuOpen_2->addAction(actionDataset_Test);
        menuOpen_2->addAction(actionDataset_Validation);
        menuSave->addAction(actionSave_dataset);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        actionDataSet->setText(QApplication::translate("MainWindow", "DataSet", Q_NULLPTR));
        actionDataset_Images_Directory_2->setText(QApplication::translate("MainWindow", "Dataset Images Directory", Q_NULLPTR));
        actionCategories_2->setText(QApplication::translate("MainWindow", "Categories", Q_NULLPTR));
        actionSave_dataset->setText(QApplication::translate("MainWindow", "Save dataset", Q_NULLPTR));
        actionDataset_Test->setText(QApplication::translate("MainWindow", "Dataset Test", Q_NULLPTR));
        actionDataset_Validation->setText(QApplication::translate("MainWindow", "Dataset Validation", Q_NULLPTR));
        groupBox_2->setTitle(QApplication::translate("MainWindow", "Sample info", Q_NULLPTR));
        label->setText(QApplication::translate("MainWindow", "True Label", Q_NULLPTR));
        label_2->setText(QApplication::translate("MainWindow", " Id", Q_NULLPTR));
        pushButton_4->setText(QApplication::translate("MainWindow", "Update info", Q_NULLPTR));
        pushButton_3->setText(QApplication::translate("MainWindow", "Save", Q_NULLPTR));
        pushButton->setText(QApplication::translate("MainWindow", "Remove", Q_NULLPTR));
        pushButton_2->setText(QApplication::translate("MainWindow", "Support Vectors", Q_NULLPTR));
        groupBox->setTitle(QApplication::translate("MainWindow", "General options", Q_NULLPTR));
        kNearestLabel->setText(QApplication::translate("MainWindow", "k-nearest:", Q_NULLPTR));
        preProcessLabel->setText(QApplication::translate("MainWindow", "Pre process", Q_NULLPTR));
        preProcessComboBox->clear();
        preProcessComboBox->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "None", Q_NULLPTR)
         << QApplication::translate("MainWindow", "PCA", Q_NULLPTR)
         << QApplication::translate("MainWindow", "LDA", Q_NULLPTR)
        );
        pca_componentsLabel->setText(QApplication::translate("MainWindow", "pca_components:", Q_NULLPTR));
        menuOpen->setTitle(QApplication::translate("MainWindow", "File", Q_NULLPTR));
        menuOpen_2->setTitle(QApplication::translate("MainWindow", "Open", Q_NULLPTR));
        menuSave->setTitle(QApplication::translate("MainWindow", "Save", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
