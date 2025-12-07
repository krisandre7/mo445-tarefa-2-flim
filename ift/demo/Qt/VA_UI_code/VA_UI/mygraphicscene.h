#ifndef MYGRAPHICSCENE_H
#define MYGRAPHICSCENE_H

#include <QObject>
#include <QWidget>
#include <QPointF>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QWheelEvent>
#include <QtGui>

class MyGraphicScene : public QGraphicsScene
{
    Q_OBJECT
public:
    explicit MyGraphicScene(QObject *parent = 0);

signals:
    void sendMouseCoordinates(const QPointF &,const QPointF &);

public slots:

private:
    QPointF firstMouseLocation;
    QPointF lastMouseLocation;
protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* ev);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* ev);
    void mouseWheelEvent(QWheelEvent* event);
};

#endif // MYGRAPHICSCENE_H
