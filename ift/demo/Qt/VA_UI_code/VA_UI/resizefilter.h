#ifndef RESIZEFILTER_H
#define RESIZEFILTER_H

#include <QObject>
#include <QtGui>

class ResizeFilter : public QObject
{
    Q_OBJECT
    public:
      ResizeFilter();
    protected:
      bool eventFilter(QObject *obj, QEvent *event);
    signals:
      void windowSizeChangedSignal(QSize size);
};

#endif // RESIZEFILTER_H
