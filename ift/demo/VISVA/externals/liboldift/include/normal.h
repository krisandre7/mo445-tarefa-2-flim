#ifndef _NORMAL_H_
#define _NORMAL_H_

#include "scene.h"
#include "shell.h"
#include "geometry.h"

typedef float (*VectorFun)(Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V);

void    SetObjectNormal (Shell *shell, Scene *scn, VectorFun fun, int obj);
void    SetShellNormal (Shell *shell, Scene *scn, VectorFun fun);
void    SetBodyShellNormal (Shell *shell, Scene *scn, VectorFun fun);
void    SetDistNormal (Shell *shell);
Scene    *NormalScene (Scene *scn, VectorFun fun);

float    Intensity3    (Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V);
float    Gradient3     (Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V);
float    DistGradient3 (Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V);

ushort  GetNormalIndex (Vector *normal);
Vector *CreateNormalTable ();

#endif








