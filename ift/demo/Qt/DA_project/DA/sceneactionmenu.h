#ifndef SCENEACTIONMENU_H
#define SCENEACTIONMENU_H


#include <QAction>
#include "global.h"

class SceneActionMenu: public QAction
{
public:
    SceneActionMenu(QObject *parent = 0);
    SceneActionMenu(const QString &text, QObject *parent = 0,
                    DrawingOption _drawingOption = COLOR,
                    ObjectDrawingOption _objectDrawingOption = NONE);




    DrawingOption drawingOption;
    ObjectDrawingOption objectDrawingOption;

};

#endif // SCENEACTIONMENU_H
