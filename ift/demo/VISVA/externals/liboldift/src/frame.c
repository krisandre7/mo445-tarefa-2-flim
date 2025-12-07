#include "frame.h"
#include "common.h"

Frame     *CreateFrame(int xsize, int ysize, int zsize)
{
  Frame *fr=NULL;

  fr = (Frame *) calloc(1,sizeof(Frame));
  if (fr == NULL) 
    Error(MSG1,"CreateFrame");

  fr->xsize = xsize;
  fr->ysize = ysize;
  fr->zsize = zsize;

  SetFrameVert(fr,0,xsize-1,0,ysize-1,0,zsize-1);

  fr->face[0].vert[0] = 0;  
  fr->face[0].vert[1] = 4;  
  fr->face[0].vert[2] = 6;
  fr->face[0].vert[3] = 2;

  fr->face[1].vert[0] = 1;  
  fr->face[1].vert[1] = 3;  
  fr->face[1].vert[2] = 7;
  fr->face[1].vert[3] = 5;

  fr->face[2].vert[0] = 0;  
  fr->face[2].vert[1] = 1;  
  fr->face[2].vert[2] = 5;
  fr->face[2].vert[3] = 4;

  fr->face[3].vert[0] = 2;  
  fr->face[3].vert[1] = 6;  
  fr->face[3].vert[2] = 7;
  fr->face[3].vert[3] = 3;

  fr->face[4].vert[0] = 0;  
  fr->face[4].vert[1] = 2;  
  fr->face[4].vert[2] = 3;
  fr->face[4].vert[3] = 1;

  fr->face[5].vert[0] = 4;  
  fr->face[5].vert[1] = 5;  
  fr->face[5].vert[2] = 7;
  fr->face[5].vert[3] = 6;
  
  fr->edge[0].vert[0] = 0;  fr->edge[0].vert[1] = 2;
  fr->edge[1].vert[0] = 0;  fr->edge[1].vert[1] = 1;
  fr->edge[2].vert[0] = 1;  fr->edge[2].vert[1] = 3;
  fr->edge[3].vert[0] = 2;  fr->edge[3].vert[1] = 3;
  fr->edge[4].vert[0] = 0;  fr->edge[4].vert[1] = 4;
  fr->edge[5].vert[0] = 4;  fr->edge[5].vert[1] = 6;
  fr->edge[6].vert[0] = 2;  fr->edge[6].vert[1] = 6;
  fr->edge[7].vert[0] = 4;  fr->edge[7].vert[1] = 5;
  fr->edge[8].vert[0] = 5;  fr->edge[8].vert[1] = 7;
  fr->edge[9].vert[0] = 6;  fr->edge[9].vert[1] = 7;
  fr->edge[10].vert[0] = 3;  fr->edge[10].vert[1] = 7;
  fr->edge[11].vert[0] = 1;  fr->edge[11].vert[1] = 5;

  return(fr);
}


void DestroyFrame(Frame **fr)
{
  Frame *aux=*fr;

  if (aux != NULL) {
    free(aux);
    *fr = NULL;
  }
}

void SetFrameVert(Frame *fr, int xmin, int xmax, int ymin, \
		 int ymax, int zmin, int zmax)
{
  int i;

  fr->vert[0].x = xmin; 
  fr->vert[0].y = ymin;
  fr->vert[0].z = zmin;
  fr->vert[1].x = xmax;
  fr->vert[1].y = ymin;
  fr->vert[1].z = zmin;
  fr->vert[2].x = xmin;
  fr->vert[2].y = ymax;
  fr->vert[2].z = zmin;
  fr->vert[3].x = xmax; 
  fr->vert[3].y = ymax;
  fr->vert[3].z = zmin;
  fr->vert[4].x = xmin; 
  fr->vert[4].y = ymin;
  fr->vert[4].z = zmax;
  fr->vert[5].x = xmax;
  fr->vert[5].y = ymin;
  fr->vert[5].z = zmax;
  fr->vert[6].x = xmin;
  fr->vert[6].y = ymax;
  fr->vert[6].z = zmax;
  fr->vert[7].x = xmax; 
  fr->vert[7].y = ymax;
  fr->vert[7].z = zmax;
  for (i=0; i < 8; i++) {
    fr->vert[i].x -= (fr->xsize)/2.;
    fr->vert[i].y -= (fr->ysize)/2.;
    fr->vert[i].z -= (fr->zsize)/2.;
  }
}

