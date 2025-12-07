#ifndef CUSTOMWIDGETLIST_H
#define CUSTOMWIDGETLIST_H

//#include <QObject>
#include <QListWidget>
#include <QMouseEvent>
#include <QKeyEvent>
#include "mywidgetitem.h"

enum Type
{
    SenderType, ReceiverType
};

class MyListWidget : public QListWidget
{


    Q_OBJECT
public:
    MyListWidget(QWidget *parent = 0);
    QKeyEvent *eventKeyCopy;
    void addMyItem(MyWidgetItem* itemM);
    QColor backGroundSelected;
    QColor backGroundNotSelected;

    QVector<MyWidgetItem*> itemRefs;
    QHash<QListWidgetItem*,MyWidgetItem*> hashListWidget2MyListWidget;
    MyListWidget* buddyList;
    Type listType;
    bool allItemsHaveIcon;
    void setItemComboBoxIndex(int* indices, int vectorSize);
    void setEnableComboBoxes(bool enable);
    void clearAllData();

protected:
    void mousePressEvent(QMouseEvent *event);
    void mouseDoubleClickEvent(QMouseEvent * event);
    void keyPressEvent(QKeyEvent *event);

signals:
    void propagateLabel(int label);
    void removedIndices(QVector<int>indices);
public slots:
    void labelPropagated(int label);
};

#endif // CUSTOMWIDGETLIST_H
