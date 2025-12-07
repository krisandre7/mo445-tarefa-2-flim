#ifndef _MATHEMATICS3_H_
#define _MATHEMATICS3_H_

#include "scene.h"

Scene *Diff3(Scene *scn1, Scene *scn2);
Scene *Sum3(Scene *scn1, Scene *scn2);
Scene *Or3(Scene *scn1, Scene *scn2);
Scene *And3(Scene *scn1, Scene *scn2);
Scene *XOr3(Scene *scn1, Scene *scn2);
Scene  *Complement3(Scene *scn);
Scene  *SQRT3(Scene *scn);
Scene *Add3(Scene *scn1, int value);
Scene *Mult3(Scene *scn1, Scene *scn2);
Scene  *Abs3(Scene *scn);

void Diff3inplace(Scene *scn1, Scene *scn2);
void Sum3inplace(Scene *scn1, Scene *scn2);
void Or3inplace(Scene *scn1, Scene *scn2);
void Negate3inplace(Scene *scn);

#endif

