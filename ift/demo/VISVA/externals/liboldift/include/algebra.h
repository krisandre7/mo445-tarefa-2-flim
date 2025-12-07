#ifndef _ALGEBRA_H_
#define _ALGEBRA_H_

#include "geometry.h"

char   PAxis(Vector viewer);
char   Octant(Vector viewer); 
void   MultMatrices(float M1[4][4],float M2[4][4],float M3[4][4]);
void   TransMatrix(float M1[4][4],float M2[4][4]);
Point  RotatePoint(float M[4][4], Point p);
Point  TransformPoint(float M[4][4], Point p);

#endif
