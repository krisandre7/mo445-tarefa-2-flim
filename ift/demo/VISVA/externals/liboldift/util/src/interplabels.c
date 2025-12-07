#include "common.h"
#include "scene.h"

/* 
   Interplabels - Interpolates the scene of labels (using Nearest Neighbour)
 */


Scene *InterpNN(Scene *scn,float dx,float dy,float dz){
  int value;
  Scene *scene,*tmp;
  Voxel P,Q,R; /* previous, current, and next voxel */
  float min=(float)INT_MAX;
  float walked_dist,dist_PQ;

  /* The default voxel sizes of the input scene should be dx=dy=dz=1.0 */

  if ((scn->dx == 0.0) && (scn->dy == 0.0) && (scn->dz == 0.0)) {    
  scn->dx=1.0;
  scn->dy=1.0;
  scn->dz=1.0;
  }

  /* The default voxel sizes of the output scene should be dx=dy=dz=min(dx,dy,dz) */

  if ((dx == 0.0) || (dy == 0.0) || (dz == 0.0)) {
    if (scn->dx < min)
      min = scn->dx;
    if (scn->dy < min)
      min = scn->dy;
    if (scn->dz < min)
      min = scn->dz;
    dx = min; dy = min; dz = min;
    if (min <= 0) {
      fprintf(stderr,"Voxel distance can not be negative.\n");
      exit(-1);
    }
  }

  /* If there is no need for resampling then returns input scene */

  if ((dx == scn->dx) && (dy == scn->dy) && (dz == scn->dz)) {
    scene = CopyScene(scn);
    return (scene);
  } else {
  /* Else the working image is the input image */
    scene = scn;
  }

  /* Resample in x */

  if (dx != scn->dx) {
    tmp = CreateScene((int)((float)(scene->xsize-1)*scene->dx/dx)+1,scene->ysize,scene->zsize);
    for(Q.x=0; Q.x < tmp->xsize; Q.x++)
      for(Q.z=0; Q.z < tmp->zsize; Q.z++)
        for(Q.y=0; Q.y < tmp->ysize; Q.y++) {


            walked_dist = (float)Q.x * dx; /* the walked distance so far */

            P.x = (int)(walked_dist/scn->dx); /* P is the previous pixel in the
                                             original scene */
            P.y = Q.y;
            P.z = Q.z;

            R.x = P.x + 1; /* R is the next pixel in the original
                              image. Observe that Q is in between P
                              and R. */
            R.y = P.y;
            R.z = P.z;

            dist_PQ =  walked_dist - (float)P.x * scn->dx;  /* the distance between P and Q */

            /* interpolation: P --- dPQ --- Q ---- dPR-dPQ ---- R

               I(Q) = (I(P)*(dPR-dPQ) + I(R)*dPQ) / dPR

            */

	    if (dist_PQ < scn->dx) value=VoxelValue(scene,P);
	    else  value=VoxelValue(scene,R);

            //value = (int)(( scn->dx - dist_PQ)*(float)VoxelValue(scene,P) + dist_PQ * (float)VoxelValue(scene,R) )/scn->dx;
            tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
	  }
    scene=tmp;
  }

  /* Resample in y */

  if (dy != scn->dy) {
    tmp = CreateScene(scene->xsize, (int)(((float)scene->ysize-1.0) * scn->dy / dy)+1,scene->zsize);
    for(Q.y=0; Q.y < tmp->ysize; Q.y++)
      for(Q.z=0; Q.z < tmp->zsize; Q.z++)
          for(Q.x=0; Q.x < tmp->xsize; Q.x++) {

            walked_dist = (float)Q.y * dy;

            P.x = Q.x;
            P.y = (int)(walked_dist/scn->dy);
            P.z = Q.z;

            R.x = P.x;
            R.y = P.y + 1;
            R.z = P.z;

            dist_PQ =  walked_dist - (float)P.y * scn->dy;
	    /* comecar a adaptar daqui !! */
            
	    if (dist_PQ < scn->dy) value=VoxelValue(scene,P);
	    else  value=VoxelValue(scene,R);

            //value = (int)(( (scn->dy - dist_PQ)*(float)VoxelValue(scene,P)) + dist_PQ * (float)VoxelValue(scene,R)) / scn->dy ;
	    tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
           
          }
    if (scene != scn) {
      DestroyScene(&scene);
    }
    scene=tmp;
  }

  /* Resample in z */

  if (dz != scn->dz) {
    tmp = CreateScene(scene->xsize,scene->ysize,(int)(((float)scene->zsize-1.0) * scn->dz / dz)+1);
    for(Q.z=0; Q.z < tmp->zsize; Q.z++)
        for(Q.y=0; Q.y < tmp->ysize; Q.y++)
          for(Q.x=0; Q.x < tmp->xsize; Q.x++) {

            walked_dist = (float)Q.z * dz;

            P.x = Q.x;
            P.y = Q.y;
            P.z = (int)(walked_dist/scn->dz);

            R.x = P.x;
            R.y = P.y;
            R.z = P.z + 1;

            dist_PQ =  walked_dist - (float)P.z * scn->dz;

	    if (dist_PQ < scn->dz) value=VoxelValue(scene,P);
	    else  value=VoxelValue(scene,R);

            //value = (int)(( (scn->dz - dist_PQ)*(float)VoxelValue(scene,P)) + dist_PQ * (float)VoxelValue(scene,R)) / scn->dz ;
	    tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
	  }
    if (scene != scn) {
      DestroyScene(&scene);
    }
    scene=tmp;
  }
 
  scene->dx=dx;
  scene->dy=dy;
  scene->dz=dz;
  scene->maxval=scn->maxval;
  return(scene);
}









int main(int argc, char **argv)
{
  char filename[200];
  Scene *scn,*inter;

  /*--------------------------------------------------------*/
#ifndef _WIN32
  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;
#endif

  /*--------------------------------------------------------*/

  if (argc != 5) {
    fprintf(stderr,"usage: interp <basename> <dx> <dy> <dz>\n");
    exit(-1);
  }
  sprintf(filename,"%s.scn",argv[1]);
  scn    = ReadScene(filename);
  inter = InterpNN(scn,atof(argv[2]),atof(argv[3]),atof(argv[4])); 
  sprintf(filename,"%s_int.scn",argv[1]);
  WriteScene(inter,filename);
  DestroyScene(&scn);
  DestroyScene(&inter);

  /* ---------------------------------------------------------- */
#ifndef _WIN32
  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   
#endif

  return(0);
}
