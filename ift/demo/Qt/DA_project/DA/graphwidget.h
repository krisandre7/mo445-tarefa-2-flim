#ifndef GRAPHWIDGET_H
#define GRAPHWIDGET_H

#include <QGraphicsView>
#include <QMenu>
#include "edge.h"
#include "sceneactionmenu.h"

class Node;

class GraphWidget : public QGraphicsView
{
    Q_OBJECT

public:
    GraphWidget(QWidget *parent = 0);

    void itemMoved();
    QList<SceneActionMenu *> * actions;
    QMenu mouseContextMenu;
    QList<Node*> innerNodes;
    QList<Edge*> innerEdges;
    QHash<int, int> idToNode;
    QHash<int, int> idToEdge;

    void addNode(Node* node);
    void addNode(Node* node, int id);

    void addEdge(Edge* edge);
    void addEdge(Edge* edge, int id);

    int findEdgeId(int id);
    int findNodeId(int id);

    Node* node(int idx);
    Edge* edge(int idx);

    const QList<Node*>& nodes();
    const QList<Edge*>& edges();

    void clear();

    Node* removeNode(int idx);
    Edge* removeEdge(int idx);

public slots:
    void zoomIn();
    void zoomOut();


protected:
    void keyPressEvent(QKeyEvent *event) override;
    void timerEvent(QTimerEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;
    void contextMenuEvent(QContextMenuEvent *event);
    void mouseMoveEvent(QMouseEvent * event) ;
    void mousePressEvent(QMouseEvent *event);

    //void mouseReleaseEvent(QGraphicsSceneMouseEvent *event) override;

#ifndef QT_NO_WHEELEVENT
    void wheelEvent(QWheelEvent *event) override;
#endif
    void drawBackground(QPainter *painter, const QRectF &rect) override;

    void scaleView(qreal scaleFactor);

private:
    int timerId;
    QPointF mousePositionClick;
    QPointF released;
    QMenu* createMenu(QList<SceneActionMenu *>actionsList);
    QList<SceneActionMenu *> * createActions();
    void createColorActions(QList<SceneActionMenu *> * actionsList);
    void createHiddenActions(QList<SceneActionMenu *> * actionsList);
    void createLabelPropagationActions(QList<SceneActionMenu *>* actionsList);
    void uncheckAllDrawingOption(DrawingOption _drawingOption);

signals:
    void SelectedOptionSginal(DrawingManager drawingManager);
    void mouseCoordinates();
    void mouseMoveCoordinates(QPointF* currentMousePositionInGraphicArea,
                              QPointF* currentMousePositionInSceneCoordinates,
                              bool keepPressing);
    void mousePressed(QPointF* currentMousePositionInGraphicArea,
                      QPointF* currentMousePositionInSceneCoordinates, Qt::MouseButton buttons);
    void deleteObjectSignals();
};

#endif // GRAPHWIDGET_H
