#ifndef _MATCHING_H_
#define _MATCHING_H_

#include "scene.h"
#include "matrix.h"
#include "curve.h"
#include "radiometric3.h"
#include "math.h"
double QuadraticMeanError(Scene *scn1, Scene *scn2);
double MutualInformation(Scene *scn1, Scene *scn2);

#endif
