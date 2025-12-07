#ifndef CUSTOMGRAPHICSVIEW_H
#define CUSTOMGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QMouseEvent>
#include <QtGui>
#include <QComboBox>
#include <QEvent>
#include <QTime>
#include <QDebug>
#include <QGraphicsItem>
#include <QMenu>
#include "ift.h"
#include "mainwindow.h"

class CustomGraphicsView : public QGraphicsView
{
     Q_OBJECT
public:
    CustomGraphicsView(QWidget *parent = 0);
    ~CustomGraphicsView();
    bool autoReprojection;
    QColor sceneRectColor;
    QColor backgroundColor;

    void computeSceneScalingFactor();
    void paintNodes();
    void paintNodesByTrueClass();
    void paintNodesByPredictedClass();
    void paintNodesBySupervisedClass();
    void paintNodesByPropagatedClass();
    void paintNodesBySupervisedAndPropagatedClass();
    void paintNodesByGroup();
    void paintNodesByWeight();
    void propagateLabel(int labelIdPropagated);
    void paintNodesByGlobalkNN();

private:
    const int avaregeTimeDisplayToolTip_miliseconds = 0;
    const int avaregeTimeHumanReaction_miliseconds = 50;
    const int totalTimeReaction = avaregeTimeDisplayToolTip_miliseconds+avaregeTimeHumanReaction_miliseconds;
    int MouseX_old = 0;
    int MouseY_old = 0;
    int MouseX_new = 0;
    int MouseY_new = 0;
    int nMilliseconds;
    QTime myTimer;
    QGraphicsItem* lastItem = NULL;
    bool validStart;
    MainWindow *mainWindow = NULL;

signals:
    void mouseContextMenuRequest_signal(QContextMenuEvent *event);
    void sceneAreaSizeChanged_signal();

public slots:
    void updateSceneAutoReprojection_slot(bool autoReproject);
    void createMouseMenuForPropagation_slot(QContextMenuEvent *event);
    void updateNodesPosition_slot();
    void addNewClassToDataSet_slot();
    void addNewGroupToDataSet_slot();
    void selectClass_slot();
    void selectGroup_slot();
    void selectSample_slot();

protected:
#ifndef QT_NO_WHEELEVENT
    void wheelEvent(QWheelEvent *event) override;

#endif
    void scaleView(qreal scaleFactor);
    void drawBackground(QPainter *painter, const QRectF &rect) override;
    void mouseMoveEvent(QMouseEvent *event) ;
    void contextMenuEvent(QContextMenuEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent* event);
    void mouseDoubleClickEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
};

#endif // CUSTOMGRAPHICSVIEW_H
