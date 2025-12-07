#include "classify.h"
#include "geometry.h"

void ClassifyObject(Shell *shell, Scene *scn, Curve *c, VectorFun fun, int obj)
{  
  int i,i1,i2;
  Voxel v;
  Curve *fc;
  Vector normal;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;
  int mag;

  A = Spheric(1.8);
  V = AdjVoxels(scn,A);

  fc = FillCurve(c,1.0);

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
	    mag = (int)fun(scn,&v,&normal,A,V);
	    shell->voxel[i].opac *= fc->Y[abs(mag)];
	  }
	}        
      }
    }
  DestroyCurve(&fc);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

void ClassifyBodyShell (Shell *shell, Curve *c1, Curve *c2)
{  
  int i;
  Curve *fc1;  
  Curve *fc2;  
  int mag;
  int val;

  if (c1)
    fc1 = FillCurve(c1,1.0);
  if (c2)
    fc2 = FillCurve(c2,1.0);

  for (i=0;i<shell->nvoxels;i++) { 
    val = (int)shell->body[i].val;
    mag = (int)shell->body[i].normalmag;    
    if (c1) 
      shell->voxel[i].opac *= fc1->Y[abs(val)];
    if (c2)
      shell->voxel[i].opac *= fc2->Y[abs(mag)];
    if (shell->voxel[i].opac > 0) {
      shell->voxel[i].obj = 1;
    }
  }
  DestroyCurve(&fc1);
  DestroyCurve(&fc2);
  shell->nobjs =1;

}


void ClassifyShell (Shell *shell, Scene *scn, Curve *c, VectorFun fun)
{  
  int i,i1,i2;
  Voxel v;
  Curve *fc;
  Vector normal;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;
  int mag;

  A = Spheric(1.8);
  V = AdjVoxels(scn,A);

  fc = FillCurve(c,1.0);

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
	  mag = (int)fun(scn,&v,&normal,A,V);
	  shell->voxel[i].opac *= fc->Y[abs(mag)];
	}        
      }
    }
  DestroyCurve(&fc);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

void ClassifyScene (Scene *scn, Scene *alpha, Curve *c, VectorFun fun)
{  
  int i;
  Voxel v;
  Curve *fc;
  Vector normal;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;

  A = Spheric(1.8);
  V = AdjVoxels(scn,A);

  fc = FillCurve(c,1.0);
  i = 0;
  for (v.z = 0; v.z < scn->zsize; v.z ++) 
    for (v.y = 0; v.y < scn->ysize; v.y ++) 
      for (v.x = 0; v.x < scn->xsize; v.x ++) {
	i = scn->tbz[v.z]+scn->tby[v.y]+v.x;
	  alpha->data[i] *= fc->Y[abs((int)fun(scn,&v,&normal,A,V))];
      }
  
  DestroyCurve(&fc);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

void ClassifyTDE (Scene *tde, Scene *alpha, Scene *label, Curve *c, VectorFun fun, int obj)
{  
  int i;
  Voxel v;
  Curve *fc;
  Vector normal;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;

  A = Spheric(1.8);
  V = AdjVoxels(tde,A);

  fc = FillCurve(c,1.0);
  i = 0;
  for (v.z = 0; v.z < tde->zsize; v.z ++) 
    for (v.y = 0; v.y < tde->ysize; v.y ++) 
      for (v.x = 0; v.x < tde->xsize; v.x ++) {
	i = tde->tbz[v.z]+tde->tby[v.y]+v.x;
	if (label->data[i] == obj) {
	  alpha->data[i] *= fc->Y[abs((int)fun(tde,&v,&normal,A,V))];
	}
      }
  
  DestroyCurve(&fc);
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

void  ResetShellOpacity(Shell *shell) {
  
  int i;

  for(i=0;i<shell->nvoxels;i++)
    shell->voxel[i].opac = 255;
}

Curve *SceneTraining (Scene *scn, Curve3D *points, VectorFun fun)
{  
  Voxel v;
  Vector normal;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;
  Curve *c;
  int i;
  int min=scn->maxval,max=0;

  A = Spheric(2.5);
  V = AdjVoxels(scn,A);

  for (i=0;i<points->n;i++) {   
    v.x = (int)(points->X[i]);
    v.y = (int)(points->Y[i]);
    v.z = (int)(points->Z[i]);
    max = MAX(max,(int)(fun(scn,&v,&normal,A,V)));
    min = MIN(min,(int)(fun(scn,&v,&normal,A,V)));
  }

  c = CreateCurve(2);

  c->X[0] = min;
  c->Y[0] = 0.0;

  c->X[1] = max;
  c->Y[1] = 1.0;

  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
  return(c);
}

Curve *ShellTraining (Shell *shell, Curve3D *points)
{  
  Voxel v;
  Curve *c;
  int i,j,mag,val;
  int minmag=INT_MAX,maxmag=0;
  int minval=INT_MAX,maxval=0;
    
  for (i=0;i<points->n;i++) {   
    v.x = (int)points->X[i];
    v.y = (int)points->Y[i];
    v.z = (int)points->Z[i];
    j = VoxelExist(shell,&v);
    if (j >= 0) {
      mag = (int)shell->body[j].normalmag;
      val = (int)shell->body[j].val;
    } else {
      mag = 0;
      val = 0;
    }
    maxval = MAX(maxval,val);
    maxmag = MAX(maxval,mag);
    minval = MIN(minval,val);
    minmag = MIN(minval,mag);
  }
 

  c = CreateCurve(4);

  c->X[0] = minval;
  c->Y[0] = 0.0;
  c->X[1] = maxval;
  c->Y[1] = 1.0;

  c->X[0] = minmag;
  c->Y[0] = 0.0;
  c->X[1] = maxmag;
  c->Y[1] = 1.0;


  return(c);
}

Curve *ShellTrainingIntensity (Shell *shell, Curve3D *points)
{  
  Voxel v;
  Curve *c;
  int i,j,val;
  int min=INT_MAX,max=0,dev=0;
    
  for (i=0;i<points->n;i++) {   
    v.x = (int)points->X[i];
    v.y = (int)points->Y[i];
    v.z = (int)points->Z[i];
    j = VoxelExist(shell,&v);
    if (j >= 0) {      
      val = (int)shell->body[j].val;
    } else {
      val = 0;
    }
    max = MAX(max,val);
    min = MIN(min,val);
  }
  dev= max-min;
 
  c = CreateCurve(4);

  c->X[0] = min;
  c->Y[0] = 0.0;
  c->X[1] = min + dev*0.25;
  c->Y[1] = 1.0;
  c->X[2] = max - dev*0.25;
  c->Y[2] = 1.0;
  c->X[3] = max;
  c->Y[3] = 0.0;
  return(c);
}

Curve *ShellTrainingGradient (Shell *shell, Curve3D *points)
{  
  Voxel v;
  Curve *c;
  int i,j,val;
  int min=INT_MAX,max=0;
    
  for (i=0;i<points->n;i++) {   
    v.x = (int)points->X[i];
    v.y = (int)points->Y[i];
    v.z = (int)points->Z[i];
    j = VoxelExist(shell,&v);
    if (j >= 0) {
      val = (int)shell->body[j].normalmag;
    } else {
      val = 0;
    }
    max = MAX(max,val);
    min = MIN(min,val);
  }
 
  c = CreateCurve(3);

  c->X[0] = min;
  c->Y[0] = 0.0;
  c->X[1] = max;
  c->Y[1] = 1.0;
  c->X[2] = GetShellMaximum(shell);
  c->Y[2] = 1.0;
  return(c);
}




