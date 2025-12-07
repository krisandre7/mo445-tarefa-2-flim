/********************************************************************************
** Form generated from reading UI file 'visva.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_VISVA_H
#define UI_VISVA_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QScrollArea>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Visva
{
public:
    QWidget *centralWidget;
    QGridLayout *gridLayout_2;
    QHBoxLayout *horizontalLayout;
    QPushButton *btNewProject;
    QPushButton *btOpenProject;
    QPushButton *btSaveProject;
    QFrame *line;
    QPushButton *btTransformations;
    QPushButton *btSegmentation;
    QPushButton *btVisualization;
    QPushButton *btRegistration;
    QPushButton *btAnalisys;
    QFrame *line_2;
    QSpacerItem *horizontalSpacer;
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QComboBox *cbLabel;
    QComboBox *cbData;
    QComboBox *cbMarker;
    QLabel *lblLabel;
    QLabel *lblData;
    QLabel *lblMarker;
    QGroupBox *groupBox_3;
    QPushButton *btZoomIn;
    QPushButton *btZoomOut;
    QPushButton *btBrightContr;
    QGroupBox *groupBox_2;
    QPushButton *btObjects;
    QSpacerItem *verticalSpacer;
    QGridLayout *gridLayout;
    QScrollArea *saCoronal;
    QWidget *scrollAreaWidgetContents_2;
    QGridLayout *gridLayout_4;
    QLabel *lblCoronal;
    QWidget *widget;
    QHBoxLayout *horizontalLayout_2;
    QSpinBox *spSliceAxial;
    QPushButton *pushButton_5;
    QLabel *lblAxialX;
    QLabel *lblAxialY;
    QLabel *lblAxialZ;
    QSpacerItem *horizontalSpacer_2;
    QWidget *widget_2;
    QHBoxLayout *horizontalLayout_3;
    QSpinBox *spSliceCoronal;
    QPushButton *pushButton_6;
    QLabel *lblCoronalX;
    QLabel *lblCoronalY;
    QLabel *lblCoronalZ;
    QSpacerItem *horizontalSpacer_3;
    QScrollArea *saSagital;
    QWidget *scrollAreaWidgetContents_3;
    QGridLayout *gridLayout_5;
    QLabel *lblSagital;
    QScrollArea *saRender;
    QWidget *scrollAreaWidgetContents_4;
    QWidget *widget_4;
    QHBoxLayout *horizontalLayout_5;
    QPushButton *btRender3D;
    QPushButton *btnDisplayPlanes;
    QSpacerItem *horizontalSpacer_5;
    QWidget *widget_3;
    QHBoxLayout *horizontalLayout_4;
    QSpinBox *spSliceSagital;
    QPushButton *pushButton_7;
    QLabel *lblSagitalX;
    QLabel *lblSagitalY;
    QLabel *lblSagitalZ;
    QSpacerItem *horizontalSpacer_4;
    QScrollArea *saAxial;
    QWidget *scrollAreaWidgetContents;
    QGridLayout *gridLayout_3;
    QLabel *lblAxial;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *Visva)
    {
        if (Visva->objectName().isEmpty())
            Visva->setObjectName(QStringLiteral("Visva"));
        Visva->resize(1171, 762);
        QIcon icon;
        icon.addFile(QStringLiteral("../../../../../../Documents/Lungs.ico"), QSize(), QIcon::Normal, QIcon::Off);
        Visva->setWindowIcon(icon);
        centralWidget = new QWidget(Visva);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        gridLayout_2 = new QGridLayout(centralWidget);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        btNewProject = new QPushButton(centralWidget);
        btNewProject->setObjectName(QStringLiteral("btNewProject"));

        horizontalLayout->addWidget(btNewProject);

        btOpenProject = new QPushButton(centralWidget);
        btOpenProject->setObjectName(QStringLiteral("btOpenProject"));

        horizontalLayout->addWidget(btOpenProject);

        btSaveProject = new QPushButton(centralWidget);
        btSaveProject->setObjectName(QStringLiteral("btSaveProject"));

        horizontalLayout->addWidget(btSaveProject);

        line = new QFrame(centralWidget);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line);

        btTransformations = new QPushButton(centralWidget);
        btTransformations->setObjectName(QStringLiteral("btTransformations"));

        horizontalLayout->addWidget(btTransformations);

        btSegmentation = new QPushButton(centralWidget);
        btSegmentation->setObjectName(QStringLiteral("btSegmentation"));

        horizontalLayout->addWidget(btSegmentation);

        btVisualization = new QPushButton(centralWidget);
        btVisualization->setObjectName(QStringLiteral("btVisualization"));

        horizontalLayout->addWidget(btVisualization);

        btRegistration = new QPushButton(centralWidget);
        btRegistration->setObjectName(QStringLiteral("btRegistration"));

        horizontalLayout->addWidget(btRegistration);

        btAnalisys = new QPushButton(centralWidget);
        btAnalisys->setObjectName(QStringLiteral("btAnalisys"));

        horizontalLayout->addWidget(btAnalisys);

        line_2 = new QFrame(centralWidget);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setFrameShape(QFrame::VLine);
        line_2->setFrameShadow(QFrame::Sunken);

        horizontalLayout->addWidget(line_2);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);


        gridLayout_2->addLayout(horizontalLayout, 0, 0, 1, 2);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setSpacing(6);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setSizeConstraint(QLayout::SetDefaultConstraint);
        verticalLayout->setContentsMargins(-1, 5, -1, 5);
        groupBox = new QGroupBox(centralWidget);
        groupBox->setObjectName(QStringLiteral("groupBox"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy);
        groupBox->setMinimumSize(QSize(155, 0));
        groupBox->setMaximumSize(QSize(16777215, 150));
        groupBox->setFlat(false);
        groupBox->setCheckable(false);
        cbLabel = new QComboBox(groupBox);
        cbLabel->setObjectName(QStringLiteral("cbLabel"));
        cbLabel->setGeometry(QRect(62, 28, 85, 27));
        cbData = new QComboBox(groupBox);
        cbData->setObjectName(QStringLiteral("cbData"));
        cbData->setGeometry(QRect(46, 68, 101, 27));
        cbMarker = new QComboBox(groupBox);
        cbMarker->setObjectName(QStringLiteral("cbMarker"));
        cbMarker->setGeometry(QRect(96, 108, 51, 27));
        lblLabel = new QLabel(groupBox);
        lblLabel->setObjectName(QStringLiteral("lblLabel"));
        lblLabel->setGeometry(QRect(7, 32, 67, 20));
        lblData = new QLabel(groupBox);
        lblData->setObjectName(QStringLiteral("lblData"));
        lblData->setGeometry(QRect(7, 71, 67, 17));
        lblMarker = new QLabel(groupBox);
        lblMarker->setObjectName(QStringLiteral("lblMarker"));
        lblMarker->setGeometry(QRect(7, 113, 67, 17));

        verticalLayout->addWidget(groupBox);

        groupBox_3 = new QGroupBox(centralWidget);
        groupBox_3->setObjectName(QStringLiteral("groupBox_3"));
        sizePolicy.setHeightForWidth(groupBox_3->sizePolicy().hasHeightForWidth());
        groupBox_3->setSizePolicy(sizePolicy);
        groupBox_3->setMinimumSize(QSize(150, 0));
        groupBox_3->setMaximumSize(QSize(16777215, 70));
        btZoomIn = new QPushButton(groupBox_3);
        btZoomIn->setObjectName(QStringLiteral("btZoomIn"));
        btZoomIn->setGeometry(QRect(7, 30, 41, 27));
        QIcon icon1;
        icon1.addFile(QStringLiteral("../../../../../../Downloads/1452218513_magnifier_zoom_in.png"), QSize(), QIcon::Normal, QIcon::Off);
        btZoomIn->setIcon(icon1);
        btZoomOut = new QPushButton(groupBox_3);
        btZoomOut->setObjectName(QStringLiteral("btZoomOut"));
        btZoomOut->setGeometry(QRect(57, 30, 41, 27));
        QIcon icon2;
        icon2.addFile(QStringLiteral("../../../../../../Downloads/1452218519_magnifier_zoom_out.png"), QSize(), QIcon::Normal, QIcon::Off);
        btZoomOut->setIcon(icon2);
        btBrightContr = new QPushButton(groupBox_3);
        btBrightContr->setObjectName(QStringLiteral("btBrightContr"));
        btBrightContr->setGeometry(QRect(107, 30, 41, 27));
        QIcon icon3;
        icon3.addFile(QStringLiteral("../../../../../../Documents/Contrast_Brightness3.ico"), QSize(), QIcon::Normal, QIcon::Off);
        btBrightContr->setIcon(icon3);
        btBrightContr->setIconSize(QSize(64, 16));

        verticalLayout->addWidget(groupBox_3);

        groupBox_2 = new QGroupBox(centralWidget);
        groupBox_2->setObjectName(QStringLiteral("groupBox_2"));
        sizePolicy.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy);
        groupBox_2->setMinimumSize(QSize(150, 0));
        groupBox_2->setMaximumSize(QSize(16777215, 70));
        btObjects = new QPushButton(groupBox_2);
        btObjects->setObjectName(QStringLiteral("btObjects"));
        btObjects->setGeometry(QRect(57, 30, 41, 27));
        QIcon icon4;
        icon4.addFile(QStringLiteral("../../../../../../Downloads/1452218862_hammer_screwdriver.png"), QSize(), QIcon::Normal, QIcon::Off);
        btObjects->setIcon(icon4);

        verticalLayout->addWidget(groupBox_2);

        verticalSpacer = new QSpacerItem(20, 300, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        gridLayout_2->addLayout(verticalLayout, 1, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(2);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        gridLayout->setSizeConstraint(QLayout::SetFixedSize);
        gridLayout->setContentsMargins(2, 2, 2, 2);
        saCoronal = new QScrollArea(centralWidget);
        saCoronal->setObjectName(QStringLiteral("saCoronal"));
        saCoronal->setWidgetResizable(true);
        scrollAreaWidgetContents_2 = new QWidget();
        scrollAreaWidgetContents_2->setObjectName(QStringLiteral("scrollAreaWidgetContents_2"));
        scrollAreaWidgetContents_2->setGeometry(QRect(0, 0, 490, 287));
        gridLayout_4 = new QGridLayout(scrollAreaWidgetContents_2);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_4->setContentsMargins(2, 2, 2, 2);
        lblCoronal = new QLabel(scrollAreaWidgetContents_2);
        lblCoronal->setObjectName(QStringLiteral("lblCoronal"));
        QSizePolicy sizePolicy1(QSizePolicy::Ignored, QSizePolicy::Ignored);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(lblCoronal->sizePolicy().hasHeightForWidth());
        lblCoronal->setSizePolicy(sizePolicy1);
        QPalette palette;
        QBrush brush(QColor(60, 60, 60, 255));
        brush.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Text, brush);
        QBrush brush1(QColor(255, 255, 255, 255));
        brush1.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Base, brush1);
        QBrush brush2(QColor(0, 0, 0, 255));
        brush2.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Active, QPalette::Window, brush2);
        palette.setBrush(QPalette::Inactive, QPalette::Text, brush);
        palette.setBrush(QPalette::Inactive, QPalette::Base, brush1);
        palette.setBrush(QPalette::Inactive, QPalette::Window, brush2);
        QBrush brush3(QColor(161, 160, 159, 255));
        brush3.setStyle(Qt::SolidPattern);
        palette.setBrush(QPalette::Disabled, QPalette::Text, brush3);
        palette.setBrush(QPalette::Disabled, QPalette::Base, brush2);
        palette.setBrush(QPalette::Disabled, QPalette::Window, brush2);
        lblCoronal->setPalette(palette);
        lblCoronal->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        gridLayout_4->addWidget(lblCoronal, 0, 0, 1, 1);

        saCoronal->setWidget(scrollAreaWidgetContents_2);

        gridLayout->addWidget(saCoronal, 0, 1, 1, 1);

        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(widget->sizePolicy().hasHeightForWidth());
        widget->setSizePolicy(sizePolicy2);
        widget->setMinimumSize(QSize(0, 0));
        widget->setMaximumSize(QSize(16777215, 50));
        widget->setAutoFillBackground(false);
        horizontalLayout_2 = new QHBoxLayout(widget);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        horizontalLayout_2->setContentsMargins(2, 2, 2, 2);
        spSliceAxial = new QSpinBox(widget);
        spSliceAxial->setObjectName(QStringLiteral("spSliceAxial"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Minimum);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(spSliceAxial->sizePolicy().hasHeightForWidth());
        spSliceAxial->setSizePolicy(sizePolicy3);
        spSliceAxial->setMinimumSize(QSize(110, 0));

        horizontalLayout_2->addWidget(spSliceAxial);

        pushButton_5 = new QPushButton(widget);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));
        QIcon icon5;
        icon5.addFile(QStringLiteral("../../../../../../Downloads/1452227451_Sync.png"), QSize(), QIcon::Normal, QIcon::Off);
        pushButton_5->setIcon(icon5);

        horizontalLayout_2->addWidget(pushButton_5);

        lblAxialX = new QLabel(widget);
        lblAxialX->setObjectName(QStringLiteral("lblAxialX"));
        QSizePolicy sizePolicy4(QSizePolicy::Fixed, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(lblAxialX->sizePolicy().hasHeightForWidth());
        lblAxialX->setSizePolicy(sizePolicy4);
        lblAxialX->setMinimumSize(QSize(60, 0));

        horizontalLayout_2->addWidget(lblAxialX);

        lblAxialY = new QLabel(widget);
        lblAxialY->setObjectName(QStringLiteral("lblAxialY"));
        sizePolicy4.setHeightForWidth(lblAxialY->sizePolicy().hasHeightForWidth());
        lblAxialY->setSizePolicy(sizePolicy4);
        lblAxialY->setMinimumSize(QSize(60, 0));

        horizontalLayout_2->addWidget(lblAxialY);

        lblAxialZ = new QLabel(widget);
        lblAxialZ->setObjectName(QStringLiteral("lblAxialZ"));
        sizePolicy4.setHeightForWidth(lblAxialZ->sizePolicy().hasHeightForWidth());
        lblAxialZ->setSizePolicy(sizePolicy4);
        lblAxialZ->setMinimumSize(QSize(60, 0));

        horizontalLayout_2->addWidget(lblAxialZ);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);


        gridLayout->addWidget(widget, 1, 0, 1, 1);

        widget_2 = new QWidget(centralWidget);
        widget_2->setObjectName(QStringLiteral("widget_2"));
        sizePolicy2.setHeightForWidth(widget_2->sizePolicy().hasHeightForWidth());
        widget_2->setSizePolicy(sizePolicy2);
        widget_2->setMaximumSize(QSize(16777215, 50));
        horizontalLayout_3 = new QHBoxLayout(widget_2);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(2, 2, 2, 2);
        spSliceCoronal = new QSpinBox(widget_2);
        spSliceCoronal->setObjectName(QStringLiteral("spSliceCoronal"));
        sizePolicy3.setHeightForWidth(spSliceCoronal->sizePolicy().hasHeightForWidth());
        spSliceCoronal->setSizePolicy(sizePolicy3);
        spSliceCoronal->setMinimumSize(QSize(110, 0));
        spSliceCoronal->setMaximumSize(QSize(16777215, 16777215));

        horizontalLayout_3->addWidget(spSliceCoronal);

        pushButton_6 = new QPushButton(widget_2);
        pushButton_6->setObjectName(QStringLiteral("pushButton_6"));
        pushButton_6->setIcon(icon5);

        horizontalLayout_3->addWidget(pushButton_6);

        lblCoronalX = new QLabel(widget_2);
        lblCoronalX->setObjectName(QStringLiteral("lblCoronalX"));
        sizePolicy4.setHeightForWidth(lblCoronalX->sizePolicy().hasHeightForWidth());
        lblCoronalX->setSizePolicy(sizePolicy4);
        lblCoronalX->setMinimumSize(QSize(60, 0));

        horizontalLayout_3->addWidget(lblCoronalX);

        lblCoronalY = new QLabel(widget_2);
        lblCoronalY->setObjectName(QStringLiteral("lblCoronalY"));
        sizePolicy4.setHeightForWidth(lblCoronalY->sizePolicy().hasHeightForWidth());
        lblCoronalY->setSizePolicy(sizePolicy4);
        lblCoronalY->setMinimumSize(QSize(60, 0));

        horizontalLayout_3->addWidget(lblCoronalY);

        lblCoronalZ = new QLabel(widget_2);
        lblCoronalZ->setObjectName(QStringLiteral("lblCoronalZ"));
        sizePolicy4.setHeightForWidth(lblCoronalZ->sizePolicy().hasHeightForWidth());
        lblCoronalZ->setSizePolicy(sizePolicy4);
        lblCoronalZ->setMinimumSize(QSize(60, 0));

        horizontalLayout_3->addWidget(lblCoronalZ);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_3);


        gridLayout->addWidget(widget_2, 1, 1, 1, 1);

        saSagital = new QScrollArea(centralWidget);
        saSagital->setObjectName(QStringLiteral("saSagital"));
        saSagital->setWidgetResizable(true);
        scrollAreaWidgetContents_3 = new QWidget();
        scrollAreaWidgetContents_3->setObjectName(QStringLiteral("scrollAreaWidgetContents_3"));
        scrollAreaWidgetContents_3->setGeometry(QRect(0, 0, 490, 286));
        gridLayout_5 = new QGridLayout(scrollAreaWidgetContents_3);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        gridLayout_5->setContentsMargins(2, 2, 2, 2);
        lblSagital = new QLabel(scrollAreaWidgetContents_3);
        lblSagital->setObjectName(QStringLiteral("lblSagital"));
        sizePolicy1.setHeightForWidth(lblSagital->sizePolicy().hasHeightForWidth());
        lblSagital->setSizePolicy(sizePolicy1);
        lblSagital->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        gridLayout_5->addWidget(lblSagital, 0, 0, 1, 1);

        saSagital->setWidget(scrollAreaWidgetContents_3);

        gridLayout->addWidget(saSagital, 3, 0, 1, 1);

        saRender = new QScrollArea(centralWidget);
        saRender->setObjectName(QStringLiteral("saRender"));
        saRender->setWidgetResizable(true);
        scrollAreaWidgetContents_4 = new QWidget();
        scrollAreaWidgetContents_4->setObjectName(QStringLiteral("scrollAreaWidgetContents_4"));
        scrollAreaWidgetContents_4->setGeometry(QRect(0, 0, 490, 286));
        saRender->setWidget(scrollAreaWidgetContents_4);

        gridLayout->addWidget(saRender, 3, 1, 1, 1);

        widget_4 = new QWidget(centralWidget);
        widget_4->setObjectName(QStringLiteral("widget_4"));
        sizePolicy2.setHeightForWidth(widget_4->sizePolicy().hasHeightForWidth());
        widget_4->setSizePolicy(sizePolicy2);
        widget_4->setMaximumSize(QSize(16777215, 50));
        horizontalLayout_5 = new QHBoxLayout(widget_4);
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_5->setObjectName(QStringLiteral("horizontalLayout_5"));
        horizontalLayout_5->setContentsMargins(2, 2, 2, 2);
        btRender3D = new QPushButton(widget_4);
        btRender3D->setObjectName(QStringLiteral("btRender3D"));
        QIcon icon6;
        icon6.addFile(QStringLiteral("../../../../../../Downloads/1452228283_3d-glasses.svg"), QSize(), QIcon::Normal, QIcon::Off);
        btRender3D->setIcon(icon6);

        horizontalLayout_5->addWidget(btRender3D);

        btnDisplayPlanes = new QPushButton(widget_4);
        btnDisplayPlanes->setObjectName(QStringLiteral("btnDisplayPlanes"));
        QIcon icon7;
        icon7.addFile(QStringLiteral("../../../../../../Documents/3Dplanes.ico"), QSize(), QIcon::Normal, QIcon::Off);
        btnDisplayPlanes->setIcon(icon7);
        btnDisplayPlanes->setIconSize(QSize(18, 16));

        horizontalLayout_5->addWidget(btnDisplayPlanes);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_5->addItem(horizontalSpacer_5);


        gridLayout->addWidget(widget_4, 4, 1, 1, 1);

        widget_3 = new QWidget(centralWidget);
        widget_3->setObjectName(QStringLiteral("widget_3"));
        sizePolicy2.setHeightForWidth(widget_3->sizePolicy().hasHeightForWidth());
        widget_3->setSizePolicy(sizePolicy2);
        widget_3->setMaximumSize(QSize(16777215, 50));
        horizontalLayout_4 = new QHBoxLayout(widget_3);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QStringLiteral("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(2, 2, 2, 2);
        spSliceSagital = new QSpinBox(widget_3);
        spSliceSagital->setObjectName(QStringLiteral("spSliceSagital"));
        sizePolicy3.setHeightForWidth(spSliceSagital->sizePolicy().hasHeightForWidth());
        spSliceSagital->setSizePolicy(sizePolicy3);
        spSliceSagital->setMinimumSize(QSize(110, 0));

        horizontalLayout_4->addWidget(spSliceSagital);

        pushButton_7 = new QPushButton(widget_3);
        pushButton_7->setObjectName(QStringLiteral("pushButton_7"));
        pushButton_7->setIcon(icon5);

        horizontalLayout_4->addWidget(pushButton_7);

        lblSagitalX = new QLabel(widget_3);
        lblSagitalX->setObjectName(QStringLiteral("lblSagitalX"));
        sizePolicy4.setHeightForWidth(lblSagitalX->sizePolicy().hasHeightForWidth());
        lblSagitalX->setSizePolicy(sizePolicy4);
        lblSagitalX->setMinimumSize(QSize(60, 0));

        horizontalLayout_4->addWidget(lblSagitalX);

        lblSagitalY = new QLabel(widget_3);
        lblSagitalY->setObjectName(QStringLiteral("lblSagitalY"));
        sizePolicy4.setHeightForWidth(lblSagitalY->sizePolicy().hasHeightForWidth());
        lblSagitalY->setSizePolicy(sizePolicy4);
        lblSagitalY->setMinimumSize(QSize(60, 0));

        horizontalLayout_4->addWidget(lblSagitalY);

        lblSagitalZ = new QLabel(widget_3);
        lblSagitalZ->setObjectName(QStringLiteral("lblSagitalZ"));
        sizePolicy4.setHeightForWidth(lblSagitalZ->sizePolicy().hasHeightForWidth());
        lblSagitalZ->setSizePolicy(sizePolicy4);
        lblSagitalZ->setMinimumSize(QSize(60, 0));

        horizontalLayout_4->addWidget(lblSagitalZ);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_4);


        gridLayout->addWidget(widget_3, 4, 0, 1, 1);

        saAxial = new QScrollArea(centralWidget);
        saAxial->setObjectName(QStringLiteral("saAxial"));
        saAxial->setWidgetResizable(true);
        scrollAreaWidgetContents = new QWidget();
        scrollAreaWidgetContents->setObjectName(QStringLiteral("scrollAreaWidgetContents"));
        scrollAreaWidgetContents->setGeometry(QRect(0, 0, 490, 287));
        gridLayout_3 = new QGridLayout(scrollAreaWidgetContents);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        gridLayout_3->setContentsMargins(2, 2, 2, 2);
        lblAxial = new QLabel(scrollAreaWidgetContents);
        lblAxial->setObjectName(QStringLiteral("lblAxial"));
        lblAxial->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignTop);

        gridLayout_3->addWidget(lblAxial, 0, 0, 1, 1);

        saAxial->setWidget(scrollAreaWidgetContents);

        gridLayout->addWidget(saAxial, 0, 0, 1, 1);


        gridLayout_2->addLayout(gridLayout, 1, 1, 1, 1);

        Visva->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(Visva);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1171, 25));
        Visva->setMenuBar(menuBar);
        mainToolBar = new QToolBar(Visva);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        Visva->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(Visva);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        Visva->setStatusBar(statusBar);

        retranslateUi(Visva);

        QMetaObject::connectSlotsByName(Visva);
    } // setupUi

    void retranslateUi(QMainWindow *Visva)
    {
        Visva->setWindowTitle(QApplication::translate("Visva", "Visva", 0));
        btNewProject->setText(QApplication::translate("Visva", "New Project", 0));
        btOpenProject->setText(QApplication::translate("Visva", "Open Project", 0));
        btSaveProject->setText(QApplication::translate("Visva", "Save project", 0));
        btTransformations->setText(QApplication::translate("Visva", "Transformations", 0));
        btSegmentation->setText(QApplication::translate("Visva", "Segmentation", 0));
        btVisualization->setText(QApplication::translate("Visva", "Visualization", 0));
        btRegistration->setText(QApplication::translate("Visva", "Registration", 0));
        btAnalisys->setText(QApplication::translate("Visva", "Analisys", 0));
        groupBox->setTitle(QApplication::translate("Visva", "Slice Options", 0));
        cbLabel->clear();
        cbLabel->insertItems(0, QStringList()
         << QApplication::translate("Visva", "Fill", 0)
         << QApplication::translate("Visva", "Border", 0)
         << QApplication::translate("Visva", "Inside", 0)
         << QApplication::translate("Visva", "Off", 0)
        );
        cbData->clear();
        cbData->insertItems(0, QStringList()
         << QApplication::translate("Visva", "Original", 0)
         << QApplication::translate("Visva", "ArcWeight", 0)
         << QApplication::translate("Visva", "Fuzzy Map", 0)
         << QApplication::translate("Visva", "Object", 0)
        );
        cbMarker->clear();
        cbMarker->insertItems(0, QStringList()
         << QApplication::translate("Visva", "On", 0)
         << QApplication::translate("Visva", "Off", 0)
        );
        lblLabel->setText(QApplication::translate("Visva", "Label", 0));
        lblData->setText(QApplication::translate("Visva", "Data", 0));
        lblMarker->setText(QApplication::translate("Visva", "Marker", 0));
        groupBox_3->setTitle(QApplication::translate("Visva", "Display", 0));
        btZoomIn->setText(QString());
        btZoomOut->setText(QString());
        btBrightContr->setText(QString());
        groupBox_2->setTitle(QApplication::translate("Visva", "Objects", 0));
        btObjects->setText(QString());
        lblCoronal->setText(QString());
        pushButton_5->setText(QString());
        lblAxialX->setText(QApplication::translate("Visva", "X:", 0));
        lblAxialY->setText(QApplication::translate("Visva", "Y:", 0));
        lblAxialZ->setText(QApplication::translate("Visva", "Z:", 0));
        pushButton_6->setText(QString());
        lblCoronalX->setText(QApplication::translate("Visva", "X:", 0));
        lblCoronalY->setText(QApplication::translate("Visva", "Y:", 0));
        lblCoronalZ->setText(QApplication::translate("Visva", "Z:", 0));
        lblSagital->setText(QString());
        btRender3D->setText(QString());
        btnDisplayPlanes->setText(QString());
        pushButton_7->setText(QString());
        lblSagitalX->setText(QApplication::translate("Visva", "X:", 0));
        lblSagitalY->setText(QApplication::translate("Visva", "Y:", 0));
        lblSagitalZ->setText(QApplication::translate("Visva", "Z:", 0));
        lblAxial->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class Visva: public Ui_Visva {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_VISVA_H
