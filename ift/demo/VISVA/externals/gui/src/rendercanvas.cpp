
#include "rendercanvas.h"


RenderCanvas :: RenderCanvas(wxWindow *parent) 
  : Canvas(parent){
  rot = CreateMatrix(3,3);
  rot->val[0] = 1.0; rot->val[1] = 0.0; rot->val[2] = 0.0;
  rot->val[3] = 0.0; rot->val[4] = 1.0; rot->val[5] = 0.0;
  rot->val[6] = 0.0; rot->val[7] = 0.0; rot->val[8] = 1.0;
  rotChanged = true;
}

RenderCanvas :: ~RenderCanvas(){

}

void RenderCanvas :: RotateX(float angle){
  Matrix *Rx = CreateMatrix(3,3);
  Matrix *aux;

  Rx->val[0] = 1.0;  Rx->val[1] = 0.0;         Rx->val[2] = 0.0;
  Rx->val[3] = 0.0;  Rx->val[4] = cos(angle);  Rx->val[5] = -sin(angle);
  Rx->val[6] = 0.0;  Rx->val[7] = sin(angle);  Rx->val[8] = cos(angle);

  aux = MultMatrix(Rx, rot);
  DestroyMatrix(&rot);
  DestroyMatrix(&Rx);
  rot = aux;
  rotChanged = true;
}

void RenderCanvas :: RotateY(float angle){
  Matrix *Ry = CreateMatrix(3,3);
  Matrix *aux;

  Ry->val[0] = cos(angle);  Ry->val[1] = 0.0;  Ry->val[2] = -sin(angle);
  Ry->val[3] = 0.0;         Ry->val[4] = 1.0;  Ry->val[5] = 0.0;
  Ry->val[6] = sin(angle);  Ry->val[7] = 0.0;  Ry->val[8] = cos(angle);

  aux = MultMatrix(Ry, rot);
  DestroyMatrix(&rot);
  DestroyMatrix(&Ry);
  rot = aux;
  rotChanged = true;
}


void RenderCanvas :: Rotate(float angle, Vector axis){
  Matrix *R = CreateMatrix(3,3);
  Matrix *aux;
  float c,s;
  Vector u = axis;
  VectorNormalize(&u);
  c = cos(angle);
  s = sin(angle);
  R->val[0] = c+u.x*u.x*(1.0-c);      R->val[1] = u.x*u.y*(1.0-c)-u.z*s;  R->val[2] = u.x*u.z*(1.0-c)+u.y*s;
  R->val[3] = u.y*u.x*(1.0-c)+u.z*s;  R->val[4] = c+u.y*u.y*(1.0-c);      R->val[5] = u.y*u.z*(1.0-c)-u.x*s;
  R->val[6] = u.z*u.x*(1.0-c)-u.y*s;  R->val[7] = u.z*u.y*(1.0-c)+u.x*s;  R->val[8] = c+u.z*u.z*(1.0-c);
  aux = MultMatrix(R, rot);
  DestroyMatrix(&rot);
  DestroyMatrix(&R);
  rot = aux;
  rotChanged = true;
}


