#ifndef CLICKABLELABEL_H
#define CLICKABLELABEL_H


#include <QWidget>
#include <QLabel>
#include <QPoint>
#include <QMouseEvent>


class ClickableLabel : public QLabel
{
Q_OBJECT
public:
    explicit ClickableLabel( QWidget* parent=0 );
    ~ClickableLabel();

signals:
    void mousePress(const QPoint&);
    void mouseHolding(const QPoint&);
    void mouseRelease(const QPoint&);

protected:
    void mousePressEvent(QMouseEvent* ev);
    void mouseReleaseEvent(QMouseEvent* ev);
    void mouseMoveEvent(QMouseEvent* ev);
};

#endif // CLICKABLELABEL_H
