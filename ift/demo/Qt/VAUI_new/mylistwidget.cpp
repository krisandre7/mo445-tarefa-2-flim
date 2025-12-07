#include "mylistwidget.h"
#include <QDebug>

MyListWidget::MyListWidget(QWidget *parent)
    : QListWidget(parent)
{
    //setDragDropMode(QAbstractItemView::DragDrop);
    //setDefaultDropAction(Qt::MoveAction);
    //setAlternatingRowColors(true);
    backGroundSelected = QColor(0,0,255,50);
    backGroundNotSelected = QColor(255,255,255,0);
    listType = SenderType;
    allItemsHaveIcon = false;
    setStyleSheet("border: 1px solid black");
    setViewMode(QListWidget::IconMode);
    setResizeMode(QListWidget::Adjust);
    setSelectionMode(QListWidget::ExtendedSelection);
}


void MyListWidget::addMyItem(MyWidgetItem* itemM){
    itemM->_isSelected = false;
    itemM->index = count();
    QListWidgetItem* item = new QListWidgetItem(this);
    addItem(item);
    setItemWidget(item,itemM);
    item->setSizeHint(QSize(itemIconSize_height+20,itemIconSize_width+40));
    itemRefs.push_back(itemM);
}

void MyListWidget::clearAllData(){
    clear();
    itemRefs.clear();
}

void MyListWidget::setItemComboBoxIndex(int* indices, int vectorSize){
    if(itemRefs.size() == 0 || vectorSize > itemRefs.size()){
        return;
    }
    for (int index = 0; index < vectorSize; ++index) {
        int comboItemIndex = indices[index];
        itemRefs.at(index)->myComboBox->setCurrentIndex(comboItemIndex);
    }
}

void MyListWidget::setEnableComboBoxes(bool enable){

    for (int index = 0; index < itemRefs.size(); ++index) {
        itemRefs.at(index)->myComboBox->setEnabled(enable);
    }
}

void MyListWidget::mousePressEvent(QMouseEvent *event){
    QListWidgetItem *itemObjectRef = itemAt(event->pos());
    if(listType == SenderType){
        if(Qt::ControlModifier == event->modifiers() || Qt::ShiftModifier == event->modifiers()){
            if(itemObjectRef == NULL){
                return;
            }
            //MyWidgetItem* myItem = hashListWidget2MyListWidget[itemObjectRef];
            MyWidgetItem* myItem = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObjectRef));
            if(myItem->_isSelected == true){
                itemObjectRef->setBackground(backGroundNotSelected);
                itemObjectRef->setSelected(false);
                myItem->_isSelected = false;
            }else{
                itemObjectRef->setBackground(backGroundSelected);
                itemObjectRef->setSelected(true);
                myItem->_isSelected = true;
                int indexCurrentSelected = 0;
                int lastSelected = -1;
                for (int i = 0; i < count(); ++i) {
                    QListWidgetItem* itemObject = this->item(i);
                    MyWidgetItem* myItemAux = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
                    if(myItemAux == myItem){
                        indexCurrentSelected = i;
                        break;
                    }
                    if(myItemAux->_isSelected == true){
                        lastSelected = i;
                    }
                }
                if(lastSelected == -1){
                    return;
                }
                for (int i = lastSelected; i < indexCurrentSelected; ++i) {
                    QListWidgetItem* itemObject = this->item(i);
                    MyWidgetItem* myItemAux = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
                    itemObject->setBackground(backGroundSelected);
                    itemObject->setSelected(true);
                    myItemAux->_isSelected = true;
                }
            }
            return;
        }
        for (int i=0; i< count(); i++)
        {
            QListWidgetItem* itemObject = this->item(i);
            if(itemObject == itemObjectRef){
                itemObject->setBackgroundColor(backGroundSelected);
                itemObject->setSelected(true);
                MyWidgetItem* myItem = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
                //MyWidgetItem* myItem = hashListWidget2MyListWidget[itemObject];
                myItem->_isSelected = true;
            }else{
                itemObject->setBackgroundColor(backGroundNotSelected);
                itemObject->setSelected(false);
                MyWidgetItem* myItem = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
                //MyWidgetItem* myItem = hashListWidget2MyListWidget[itemObject];
                myItem->_isSelected = false;
            }
        }
    }else if(listType == ReceiverType){
        if(itemObjectRef == NULL){
            return;
        }
        MyWidgetItem* myitemObjectRef = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObjectRef));
        int currentIndex = myitemObjectRef->myComboBox->currentIndex();
        emit propagateLabel(currentIndex);
    }
}

void MyListWidget::labelPropagated(int label){
    for (int i = 0; i < count(); i++) {
        QListWidgetItem* itemObject = this->item(i);
        MyWidgetItem* myitemObjectRef = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
        if(myitemObjectRef->_isSelected){
            myitemObjectRef->myComboBox->setCurrentIndex(label);
        }
    }
}

void MyListWidget::keyPressEvent(QKeyEvent *event){

    if(Qt::ControlModifier == event->modifiers() && event->key() == Qt::Key_A){
        for (int i=0; i< count(); i++)
        {
            QListWidgetItem* itemObject = this->item(i);
            itemObject->setBackgroundColor(QColor(0,0,255,50));
            itemObject->setSelected(true);
            MyWidgetItem* myItem = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
            myItem->_isSelected = true;
        }
    }else if(event->key() == Qt::Key_Delete){
        if(listType == ReceiverType){
            return;
        }
        QVector<int>removedIndicesVec;
        blockSignals(true);
        for (int i = count()-1; i >= 0; --i) {
            QListWidgetItem* itemObject = this->item(i);
            MyWidgetItem* myitemObjectRef = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObject));
            if(myitemObjectRef->_isSelected){
                removedIndicesVec << i;
                takeItem(i);
                itemRefs.remove(i);
            }
        }
        blockSignals(false);
        emit removedIndices(removedIndicesVec);
    }
}

void MyListWidget::mouseDoubleClickEvent(QMouseEvent * event){
    QListWidgetItem *itemObjectRef = itemAt(event->pos());
    if(itemObjectRef == NULL){
        return;
    }
    MyWidgetItem* myitemtRef = dynamic_cast<MyWidgetItem*>(this->itemWidget(itemObjectRef));
    QPixmap image(myitemtRef->path2Image);
    QLabel *labelAux = new QLabel(this);
    labelAux->setPixmap(image);
    labelAux->setWindowFlags(Qt::Window);
    labelAux->setAttribute(Qt::WA_DeleteOnClose);
    //labelAux->setWindowTitle();
    labelAux->show();
}


