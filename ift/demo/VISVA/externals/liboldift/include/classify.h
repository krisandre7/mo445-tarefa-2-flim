#ifndef _CLASSIFY_H_
#define _CLASSIFY_H_

#include "shell.h"
#include "curve.h"
#include "scene.h"
#include "normal.h"

void ClassifyObject(Shell *shell, Scene *scn, Curve *c, VectorFun fun, int obj);
void ClassifyShell (Shell  *shell, Scene *scn, Curve *c, VectorFun fun);
void ClassifyBodyShell (Shell *shell, Curve *c1, Curve *c2);
void ClassifyScene (Scene *scn, Scene  *alpha, Curve *c, VectorFun fun);
void ClassifyTDE (Scene *tde, Scene  *alpha, Scene *label, Curve *c, VectorFun fun, int obj);
void  ResetShellOpacity(Shell *shell);
Curve *SceneTraining (Scene *scn, Curve3D *points, VectorFun fun);
Curve *ShellTraining (Shell *shell, Curve3D *points);
Curve *ShellTrainingIntensity (Shell *shell, Curve3D *points);
Curve *ShellTrainingGradient (Shell *shell, Curve3D *points);


#endif


































































































