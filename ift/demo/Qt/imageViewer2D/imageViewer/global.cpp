#include "global.h"

iftImage *image2D = NULL;
iftImage *image3D = NULL;
iftImage *image3DProjection = NULL;
iftImage *image3D_labels = NULL;
iftMatrix *image3D_gradients = NULL;

float viewerDistanceX = 0;
float viewerDistanceY = 0;
float viewerDistanceZ = 0;
float lastThetaX = 0;
float lastThetaY = 0;
float thetaXaccumulated = 0;
float thetaYaccumulated = 0;
float mouseX_start = 0;
float mouseY_start = 0;
float mouseX_Delta = 0;
float mouseY_Delta = 0;
bool isMouseEvent = false;

int renderIterator = 0;

