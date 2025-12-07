#ifndef GLOBAL_H
#define GLOBAL_H


#include "ift.h"

//global variable to mainwindow
extern iftImage *image2D;
extern iftImage *image3D;
extern iftImage *image3DProjection;
extern iftImage *image3D_labels;
extern iftMatrix *image3D_display;
extern float viewerDistanceX;
extern float viewerDistanceY;
extern float viewerDistanceZ;
extern float lastThetaX;
extern float lastThetaY;
extern float thetaXaccumulated;
extern float thetaYaccumulated;
extern float mouseX_start;
extern float mouseY_start;
extern float mouseX_Delta;
extern float mouseY_Delta;
extern bool isMouseEvent;



extern int renderIterator;


#endif // GLOBAL_H
