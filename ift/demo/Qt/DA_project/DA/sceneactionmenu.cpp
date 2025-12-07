#include "sceneactionmenu.h"

SceneActionMenu::SceneActionMenu(QObject *parent) : QAction(parent)
{
    drawingOption = COLOR;
    objectDrawingOption = NONE;

}

SceneActionMenu::SceneActionMenu(const QString &text,QObject *parent,
                                 DrawingOption _drawingOption,
                                 ObjectDrawingOption _objectDrawingOption) : QAction(text,parent)
{
    drawingOption = _drawingOption;
    objectDrawingOption = _objectDrawingOption;
}
