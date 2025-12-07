#ifndef _GEOMETRIC3_H_
#define _GEOMETRIC3_H_

#include "scene.h"
#include "geometry.h"
#include "algebra.h"

void inversa(float M[4][4], float IM[4][4]);
float Cofator(float M[4][4], int l, int c);
void translacao(float T[4][4], float dx, float dy, float dz);
void RotX(float Rx[4][4], float thx);
void RotY(float Ry[4][4], float thy);
void RotZ(float Ry[4][4], float thz);
void transformScene(Scene *scn, float IM[4][4], Scene *scn_out);
void transformacaoDireta(Scene *scn, float M[4][4], Scene *scn_out);
void transformacaoMasc(Scene *scn, Scene *masc, float M[4][4], Scene *scn_out);
void limiar(Scene *scn, int thr);
void transformScene_bin(Scene *scn,float T[4][4],Scene *scn_out);
void Rotacao(float th_x,float th_y,float th_z, float T[4][4],Scene *scn,Scene *out);
void RotTrans(float th_x,float th_y,float th_z,float dx,float dy,float dz, float T[4][4],Scene *scn,Scene *out);

//void Translacao(float dx,float dy,float dz, float T[4][4],Scene *scn,Scene *out);
#endif
