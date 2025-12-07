/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QAction *actionOpen_Image;
    QAction *actionSave_View;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QFrame *upperFrame;
    QHBoxLayout *horizontalLayout_2;
    QGraphicsView *msskelView;
    QGraphicsView *thresholdView;
    QFrame *lowerFrame;
    QHBoxLayout *horizontalLayout;
    QSlider *skelSlider;
    QCheckBox *reconstCheckBox;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(792, 490);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(MainWindow->sizePolicy().hasHeightForWidth());
        MainWindow->setSizePolicy(sizePolicy);
        actionOpen_Image = new QAction(MainWindow);
        actionOpen_Image->setObjectName(QString::fromUtf8("actionOpen_Image"));
        actionSave_View = new QAction(MainWindow);
        actionSave_View->setObjectName(QString::fromUtf8("actionSave_View"));
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        upperFrame = new QFrame(centralWidget);
        upperFrame->setObjectName(QString::fromUtf8("upperFrame"));
        upperFrame->setFrameShape(QFrame::StyledPanel);
        upperFrame->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(upperFrame);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        msskelView = new QGraphicsView(upperFrame);
        msskelView->setObjectName(QString::fromUtf8("msskelView"));

        horizontalLayout_2->addWidget(msskelView);

        thresholdView = new QGraphicsView(upperFrame);
        thresholdView->setObjectName(QString::fromUtf8("thresholdView"));

        horizontalLayout_2->addWidget(thresholdView);


        verticalLayout->addWidget(upperFrame);

        lowerFrame = new QFrame(centralWidget);
        lowerFrame->setObjectName(QString::fromUtf8("lowerFrame"));
        lowerFrame->setFrameShape(QFrame::StyledPanel);
        lowerFrame->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(lowerFrame);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        skelSlider = new QSlider(lowerFrame);
        skelSlider->setObjectName(QString::fromUtf8("skelSlider"));
        skelSlider->setMinimum(1);
        skelSlider->setSliderPosition(50);
        skelSlider->setOrientation(Qt::Horizontal);

        horizontalLayout->addWidget(skelSlider);

        reconstCheckBox = new QCheckBox(lowerFrame);
        reconstCheckBox->setObjectName(QString::fromUtf8("reconstCheckBox"));

        horizontalLayout->addWidget(reconstCheckBox);


        verticalLayout->addWidget(lowerFrame);

        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 792, 22));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        MainWindow->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionOpen_Image);
        menuFile->addAction(actionSave_View);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "skel-viz", nullptr));
        actionOpen_Image->setText(QApplication::translate("MainWindow", "Open Image", nullptr));
#ifndef QT_NO_SHORTCUT
        actionOpen_Image->setShortcut(QApplication::translate("MainWindow", "Ctrl+O", nullptr));
#endif // QT_NO_SHORTCUT
        actionSave_View->setText(QApplication::translate("MainWindow", "Save View", nullptr));
#ifndef QT_NO_SHORTCUT
        actionSave_View->setShortcut(QApplication::translate("MainWindow", "Ctrl+S", nullptr));
#endif // QT_NO_SHORTCUT
        reconstCheckBox->setText(QApplication::translate("MainWindow", "Reconstruction", nullptr));
        menuFile->setTitle(QApplication::translate("MainWindow", "File", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
