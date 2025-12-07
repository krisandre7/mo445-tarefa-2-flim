/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>
#include "qcustomplot.h"

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionDataSet;
    QAction *actionDataset_Images_Directory_2;
    QAction *actionCategories_2;
    QAction *actionSave_dataset;
    QWidget *centralWidget;
    QPushButton *pushButton_2;
    QTabWidget *tabWidget;
    QWidget *tab;
    QTableWidget *tableWidgetSelectedSamples;
    QWidget *tab_2;
    QCustomPlot *widgetPlot;
    QLabel *labelImageCurrentSample;
    QListWidget *listWidgetImagesThumb;
    QGraphicsView *graphicsViewProjectionArea;
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
        MainWindow->resize(984, 666);
        MainWindow->setAutoFillBackground(false);
        actionDataSet = new QAction(MainWindow);
        actionDataSet->setObjectName(QStringLiteral("actionDataSet"));
        actionDataset_Images_Directory_2 = new QAction(MainWindow);
        actionDataset_Images_Directory_2->setObjectName(QStringLiteral("actionDataset_Images_Directory_2"));
        actionCategories_2 = new QAction(MainWindow);
        actionCategories_2->setObjectName(QStringLiteral("actionCategories_2"));
        actionSave_dataset = new QAction(MainWindow);
        actionSave_dataset->setObjectName(QStringLiteral("actionSave_dataset"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(centralWidget->sizePolicy().hasHeightForWidth());
        centralWidget->setSizePolicy(sizePolicy);
        pushButton_2 = new QPushButton(centralWidget);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));
        pushButton_2->setGeometry(QRect(700, 410, 80, 20));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        tabWidget->setGeometry(QRect(430, 10, 360, 400));
        tab = new QWidget();
        tab->setObjectName(QStringLiteral("tab"));
        tableWidgetSelectedSamples = new QTableWidget(tab);
        tableWidgetSelectedSamples->setObjectName(QStringLiteral("tableWidgetSelectedSamples"));
        tableWidgetSelectedSamples->setGeometry(QRect(0, 0, 360, 380));
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QStringLiteral("tab_2"));
        widgetPlot = new QCustomPlot(tab_2);
        widgetPlot->setObjectName(QStringLiteral("widgetPlot"));
        widgetPlot->setGeometry(QRect(0, 0, 360, 380));
        tabWidget->addTab(tab_2, QString());
        labelImageCurrentSample = new QLabel(centralWidget);
        labelImageCurrentSample->setObjectName(QStringLiteral("labelImageCurrentSample"));
        labelImageCurrentSample->setGeometry(QRect(10, 430, 141, 111));
        sizePolicy.setHeightForWidth(labelImageCurrentSample->sizePolicy().hasHeightForWidth());
        labelImageCurrentSample->setSizePolicy(sizePolicy);
        labelImageCurrentSample->setAutoFillBackground(false);
        labelImageCurrentSample->setStyleSheet(QLatin1String("background-color: rgb(255, 255, 255);\n"
""));
        listWidgetImagesThumb = new QListWidget(centralWidget);
        listWidgetImagesThumb->setObjectName(QStringLiteral("listWidgetImagesThumb"));
        listWidgetImagesThumb->setGeometry(QRect(148, 430, 651, 111));
        listWidgetImagesThumb->setAutoScroll(true);
        listWidgetImagesThumb->setAutoScrollMargin(16);
        listWidgetImagesThumb->setSelectionMode(QAbstractItemView::ExtendedSelection);
        listWidgetImagesThumb->setVerticalScrollMode(QAbstractItemView::ScrollPerItem);
        listWidgetImagesThumb->setHorizontalScrollMode(QAbstractItemView::ScrollPerItem);
        listWidgetImagesThumb->setResizeMode(QListView::Fixed);
        graphicsViewProjectionArea = new QGraphicsView(centralWidget);
        graphicsViewProjectionArea->setObjectName(QStringLiteral("graphicsViewProjectionArea"));
        graphicsViewProjectionArea->setGeometry(QRect(10, 10, 420, 420));
        graphicsViewProjectionArea->setContextMenuPolicy(Qt::CustomContextMenu);
        graphicsViewProjectionArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
        graphicsViewProjectionArea->setDragMode(QGraphicsView::RubberBandDrag);
        graphicsViewProjectionArea->setCacheMode(QGraphicsView::CacheBackground);
        graphicsViewProjectionArea->setTransformationAnchor(QGraphicsView::NoAnchor);
        graphicsViewProjectionArea->setResizeAnchor(QGraphicsView::NoAnchor);
        graphicsViewProjectionArea->setViewportUpdateMode(QGraphicsView::MinimalViewportUpdate);
        graphicsViewProjectionArea->setRubberBandSelectionMode(Qt::IntersectsItemShape);
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 984, 22));
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
        menuSave->addAction(actionSave_dataset);

        retranslateUi(MainWindow);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        actionDataSet->setText(QApplication::translate("MainWindow", "DataSet...", Q_NULLPTR));
        actionDataset_Images_Directory_2->setText(QApplication::translate("MainWindow", "Dataset Images Directory...", Q_NULLPTR));
        actionCategories_2->setText(QApplication::translate("MainWindow", "Categories...", Q_NULLPTR));
        actionSave_dataset->setText(QApplication::translate("MainWindow", "Save dataset", Q_NULLPTR));
        pushButton_2->setText(QApplication::translate("MainWindow", "Next", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("MainWindow", "Selected Samples", Q_NULLPTR));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("MainWindow", "Metrics", Q_NULLPTR));
        labelImageCurrentSample->setText(QString());
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
