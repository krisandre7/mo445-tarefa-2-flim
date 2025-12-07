#include "normal.h"
#include "adjacency3.h"
#include "analysis3.h"
#include "common.h"
#include "geometry.h"


void SetObjectNormal (Shell *shell, Scene *scn, VectorFun fun, int obj)
{  
  int i,i1,i2;
  Voxel v;
  Vector normal;
  AdjRel3 *A;
  AdjVxl *V;

  A = Spheric(5.0);
  V = AdjVoxels(scn,A);
  
  for (v.z = 0; v.z != shell->zsize; v.z ++) 
    for (v.y = 0; v.y != shell->ysize; v.y ++) {
      i = shell->Myz->tbrow[v.z] + v.y;     
      if (shell->Myz->val[i] >= 0) {	
	i1 = shell->Myz->val[i];
	i++;
	while (shell->Myz->val[i] < 0) {
	  i++;
	} 
	i2 = shell->Myz->val[i];
	for (i = i1; i != i2; i++) {
	  if (shell->voxel[i].obj == obj) {
	    v.x = shell->voxel[i].x;
	    fun(scn,&v,&normal,A,V);
	    VectorNormalize(&normal);
	    shell->voxel[i].normal = GetNormalIndex(&normal);
	  }
	}        
      }
    }
  
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

void SetShellNormal (Shell *shell, Scene *scn, VectorFun fun)
{  
  int i,i1,i2;
  Voxel v;
  Vector normal;
  AdjRel3 *A;
  AdjVxl *V;

  if (shell->body) {
    SetBodyShellNormal(shell,scn,fun);
  } else {
    
  A = Spheric(5.0);
  V = AdjVoxels(scn,A);
  
  for (v.z = 0; v.z != shell->zsize; v.z ++) 
    for (v.y = 0; v.y != shell->ysize; v.y ++) {
      i = shell->Myz->tbrow[v.z] + v.y;     
      if (shell->Myz->val[i] >= 0) {	
	i1 = shell->Myz->val[i];
	i++;
	while (shell->Myz->val[i] < 0) {
	  i++;
	} 
	i2 = shell->Myz->val[i];
	for (i = i1; i != i2; i++) {	  
	  v.x = shell->voxel[i].x;
	  fun(scn,&v,&normal,A,V);
          VectorNormalize(&normal);
	  shell->voxel[i].normal = GetNormalIndex(&normal);
	}        
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

void SetDistNormal (Shell *shell) {

  int i;
  Scene *dist,*tmp;
  AdjRel3 *A = Spheric(1.8);

  for (i=1;i<=shell->nobjs;i++) {
    tmp = ObjectScene(shell,i);
    dist = DistTrans3(tmp,A,BOTH,0);
    SetObjectNormal(shell,dist,DistGradient3,i);    
    DestroyScene(&dist);
    DestroyScene(&tmp);
  }
  DestroyAdjRel3(&A);
}

void SetBodyShellNormal (Shell *shell, Scene *scn, VectorFun fun)
{  
  int i,i1,i2;//,j=0;
  Voxel v;
  Vector normal;
  AdjRel3 *A;
  AdjVxl *V;
  int mag;

  if (shell->body == NULL) {
    Error(MSG2,"SetBodyShellNormal");
  }

  A = Spheric(2.5);
  V = AdjVoxels(scn,A);

  
  for (v.z = 0; v.z != shell->zsize; v.z ++) 
    for (v.y = 0; v.y != shell->ysize; v.y ++) {
      i = shell->Myz->tbrow[v.z] + v.y;     
      if (shell->Myz->val[i] >= 0) {	
	i1 = shell->Myz->val[i];
	i++;
	while (shell->Myz->val[i] < 0) {
	  i++;
	} 
	i2 = shell->Myz->val[i];
	for (i = i1; i != i2; i++) {	  
	  v.x = shell->voxel[i].x;
	  mag = fun(scn,&v,&normal,A,V);
          VectorNormalize(&normal);
	  //	  j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	  shell->voxel[i].normal =  GetNormalIndex(&normal); 
	  shell->body[i].normalmag = mag;	  
	}        
      }
    }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}


Scene *NormalScene(Scene *scn, VectorFun fun) {


  int i;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;
  Voxel v;
  Vector normal;
  Scene *nscn;
  nscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);

  A = Spheric(5.0);
  V = AdjVoxels(scn,A);

  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.y=0;v.y<scn->ysize;v.y++)
      for (v.x=0;v.x<scn->xsize;v.x++) {	
	normal.x = normal.y = normal.z = 0.0;	
	i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
        if (scn->data[i] != 0) {
	  fun(scn,&v,&normal,A,V);
          VectorNormalize(&normal);
	  nscn->data[i] = (int)GetNormalIndex(&normal);
	}
      }   
	    	
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
  return(nscn);
}


float Intensity3(Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V)
{
  return (float)(scn->data[scn->tbz[v->z] + scn->tby[v->y] + v->x]);
}


float Gradient3(Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V)
{

  int k,i,ivalue;

  normal->x = normal->y = normal->z = 0.0;

  k = scn->tbz[v->z] + scn->tby[v->y] + v->x;

  for (i=0;i<V->n;i++) {
    if (ValidVoxel(scn, v->x+A->dx[i], v->y+A->dy[i], v->z+A->dz[i])) {
      ivalue = scn->data[k] - scn->data[k + V->dp[i]];      
    } else {
      ivalue = scn->data[k];
    }

    if (A->dx[i])        
      normal->x += (float)ivalue * (float)A->dx[i];

    if (A->dy[i])        
      normal->y += (float)ivalue * (float)A->dy[i];

    if (A->dz[i])        
      normal->z += (float)ivalue * (float)A->dz[i];
  }
  return VectorMagnitude(normal)/(float)A->n;
}

float DistGradient3(Scene *scn, Voxel *v, Vector *normal, AdjRel3 *A, AdjVxl *V){

  int k,i,ivalue;
  double fvalue;

  normal->x = normal->y = normal->z = 0.0;

  k = scn->tbz[v->z] + scn->tby[v->y] + v->x;

  for (i=0;i<V->n;i++) {
    if (ValidVoxel(scn, v->x+A->dx[i], v->y+A->dy[i], v->z+A->dz[i])) {
      ivalue = scn->data[k + V->dp[i]];      
      if (ivalue < 0) {
	fvalue = sqrt((double)scn->data[k]) + sqrt((double)(-ivalue));
      } else {
	fvalue = sqrt((double)scn->data[k]) - sqrt((double)( ivalue));
      }
    } else {
      fvalue = sqrt((double)scn->data[k]);
    }
    if (A->dx[i])        
      normal->x += fvalue * (float)A->dx[i];

    if (A->dy[i])        
      normal->y += fvalue * (float)A->dy[i];

    if (A->dz[i])        
      normal->z += fvalue * (float)A->dz[i];
  }
  return VectorMagnitude(normal)/(float)A->n;;
}



ushort GetNormalIndex(Vector *normal) {

  int gamma,alpha;
  ushort index;
  
  if ((normal->x == 0.0) && (normal->y == 0.0) && (normal->z == 0.0)) {
    return(0);
  }  
  gamma = (int)(asin(normal->z)*180.0/PI); /* [-90,90] */ 
  alpha = (int)(atan2(normal->y,normal->x)*180.0/PI); /* [-180,180] */
  if (alpha < 0)
    alpha += 360;
  index = ((gamma+90)*360) + alpha + 1;
     
  return(index);
}

Vector *CreateNormalTable() {

  int i,lin,col;
  float gamma,alpha;
  Vector *normaltable;

  /* creates normal look-up table */
  normaltable = (Vector*)calloc(65161,sizeof(Vector));

  if (normaltable == NULL) {
    Error(MSG1,"CreateNormalTable");
  } 
  
  normaltable[0].x = 0.0;
  normaltable[0].y = 0.0;
  normaltable[0].z = 0.0;

  i=1;
  for (lin=-90; lin <= 90; lin++){
    gamma = (PI*lin)/180.0;
    for (col=0; col < 360; col++){
      alpha = (PI*col)/180.0;
      normaltable[i].x = cos(gamma)*cos(alpha);
      normaltable[i].y = cos(gamma)*sin(alpha);
      normaltable[i].z = sin(gamma);
      i++;
    }
  }
  return normaltable;
}
