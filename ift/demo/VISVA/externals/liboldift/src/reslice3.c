#include <reslice3.h>

void ResliceX(Scene **scn, char rotation) {

  Scene *tmp=NULL;
  int x, y, z, i, j, k, p, q;

  if (*scn == NULL) {
    printf("[ResliceX] Impossible to reslice\n");
    return;
  }

  tmp = CopyScene(*scn);

  DestroyScene(scn);
  *scn = CreateScene(tmp->xsize, tmp->zsize, tmp->ysize);

  (*scn)->dx = tmp->dx;
  (*scn)->dy = tmp->dz;
  (*scn)->dz = tmp->dy;

  x = (*scn)->xsize;
  y = (*scn)->ysize;
  z = (*scn)->zsize;
  
  for (k = 0; k < z; k++) {
    for (j = 0; j < y; j++) {
      for (i = 0; i < x; i++) {
        p = i + (*scn)->tby[j] + (*scn)->tbz[ROTATION(z, k, rotation)];
        q = i + tmp->tby[k] + tmp->tbz[y - j - 1];
        (*scn)->data[p] = tmp->data[q];
      }
    }
  }
  
  DestroyScene(&tmp);  
  
}

void ResliceY(Scene **scn, char rotation) {

  Scene *tmp=NULL;
  int x, y, z, i, j, k, p, q;

  if (*scn == NULL) {
    printf("[ResliceY] Impossible to reslice\n");
    return;
  }

  tmp = CopyScene(*scn);

  DestroyScene(scn);
  *scn = CreateScene(tmp->zsize, tmp->ysize, tmp->xsize);

  (*scn)->dx = tmp->dz;
  (*scn)->dy = tmp->dy;
  (*scn)->dz = tmp->dx;

  x = (*scn)->xsize;
  y = (*scn)->ysize;
  z = (*scn)->zsize;
  
  for (k = 0; k < z; k++) {
    for (j = 0; j < y; j++) {
      for (i = 0; i < x; i++) {
        p = i + (*scn)->tby[j] + (*scn)->tbz[k];
        q = ROTATION(z, k, rotation) + tmp->tby[j] + tmp->tbz[i];
        (*scn)->data[p] = tmp->data[q];
      }
    }
  }
  
  DestroyScene(&tmp);  
  
}

void ResliceZ(Scene **scn, char rotation) {

  Scene *tmp=NULL;
  int x, y, z, i, j, k, p, q;

  if (*scn == NULL) {
    printf("[ResliceZ] Impossible to reslice\n");
    return;
  }

  tmp = CopyScene(*scn);

  DestroyScene(scn);
  *scn = CreateScene(tmp->ysize, tmp->xsize, tmp->zsize);

  (*scn)->dx = tmp->dy;
  (*scn)->dy = tmp->dx;
  (*scn)->dz = tmp->dz;

  x = (*scn)->xsize;
  y = (*scn)->ysize;
  z = (*scn)->zsize;
  
  for (k = 0; k < z; k++) {
    for (j = 0; j < y; j++) {
      for (i = 0; i < x; i++) {
        p = ROTATION(x, i, (!rotation)) + (*scn)->tby[j] + (*scn)->tbz[k];
        q = ROTATION(y, j, rotation) + tmp->tby[i] + tmp->tbz[k];
        (*scn)->data[p] = tmp->data[q];
      }
    }
  }
  
  DestroyScene(&tmp);  
  
}
