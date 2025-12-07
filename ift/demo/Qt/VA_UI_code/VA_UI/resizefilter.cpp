#include "resizefilter.h"

ResizeFilter::ResizeFilter() : QObject() {}

bool ResizeFilter::eventFilter(QObject *obj, QEvent *event)
{
  if (event->type() == QEvent::Resize)
  {
    QResizeEvent* resizeEv = static_cast<QResizeEvent*>(event);
    emit windowSizeChangedSignal(resizeEv->size());
    //qDebug() << resizeEv->size();
  }
  return QObject::eventFilter(obj, event);
}
