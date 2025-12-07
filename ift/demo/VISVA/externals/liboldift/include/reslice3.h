#ifndef _RESLICE3_H_
#define _RESLICE3_H_

#include <scene.h>

/* clockwise rotation */
#define CW 0

/* counter-clockwise rotation */
#define CCW 1

/* rotation degrees */
#define D_90 0
#define D_180 1

/* MACROS */
#define ROTATION(a, b, rot) ((rot) ? a - b - 1 : b)

/* Function definitions */
void ResliceX(Scene **scn, char rotation);
void ResliceY(Scene **scn, char rotation);
void ResliceZ(Scene **scn, char rotation);

/* TODO */
/* MirrorX, MirrorY, MirrorZ */

#endif
