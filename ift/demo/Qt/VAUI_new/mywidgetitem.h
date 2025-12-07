#ifndef MYWIDGETITEM_H
#define MYWIDGETITEM_H

#include <QWidget>
#include <QComboBox>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <QIcon>
#include <QLabel>
#include <QWidgetList>
#include <QtCore>
#include <QtGui>

#define itemIconSize_width 80
#define itemIconSize_height 80

class MyWidgetItem : public QWidget
{
    Q_OBJECT
public:
    explicit MyWidgetItem(QWidget *parent = NULL);
    MyWidgetItem(QWidget *parent = NULL,QComboBox *mycomboBox = NULL,QPixmap *image = NULL);
    MyWidgetItem(QPixmap image, QVector<QString> labelNames, QVector<QColor> labeColors, QWidget *parent = NULL);
    MyWidgetItem(QString path, QVector<QString> labelNames, QVector<QColor> labeColors, QWidget *parent = NULL);
    void updateImage(QString path);
    //MyWidgetItem(QWidget *parent = nullptr,QVector<QString> labelsNames = nullptr,QPixmap *image = nullptr);

    QComboBox *myComboBox;
    QLabel* labelImage;
    QString path2Image;
    QString fileName;
    bool _isSelected;
    bool hasRepresentativeImage;
    int index;
    unsigned long uniqueId;


protected:
    void mousePressEvent(QMouseEvent *event);

signals:
    void pressed(void);
public slots:
};

#endif // MYWIDGETITEM_H
