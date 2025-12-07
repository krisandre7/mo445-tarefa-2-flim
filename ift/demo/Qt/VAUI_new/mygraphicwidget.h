#ifndef MYGRAPHICWIDGET_H
#define MYGRAPHICWIDGET_H


#include <QGraphicsView>
#include <QMouseEvent>
#include <QEvent>
#include <QTime>


class MyGraphicWidget : public QGraphicsView
{
     Q_OBJECT
public:
    const int avaregeTimeDisplayToolTip_miliseconds = 0;
    const int avaregeTimeHumanReaction_miliseconds = 400;
    const int totalTimeReaction = avaregeTimeDisplayToolTip_miliseconds+avaregeTimeHumanReaction_miliseconds;
    int MouseX_old = 0;
    int MouseY_old = 0;
    int MouseX_new = 0;
    int MouseY_new = 0;
    int counter = 0;
    clock_t start, end;
    int nMilliseconds;
    MyGraphicWidget(QWidget *parent = 0);
    QTime myTimer;
    QGraphicsItem* lastItem = NULL;
    bool validStart;
    bool autoReprojection;

signals:
    void myMouseContextMenuSignal(QContextMenuEvent *event);
    void sceneAreaSizeChanged();

public slots:
    void updateSceneAutoReprojection(bool autoReproject);

protected:
#ifndef QT_NO_WHEELEVENT
    void wheelEvent(QWheelEvent *event) override;
#endif
    void scaleView(qreal scaleFactor);
    void drawBackground(QPainter *painter, const QRectF &rect) override;
    void mouseMoveEvent(QMouseEvent * event) ;
    void contextMenuEvent(QContextMenuEvent *event);
    void mousePressEvent(QMouseEvent *event);
    void resizeEvent(QResizeEvent* event);
    //void mousePressEvent(QMouseEvent *event);



};

#endif // MYGRAPHICWIDGET_H
