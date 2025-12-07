

#include <oldift.h>
#include "msp.h"
#include "arraylist.h"
#include "segmobject.h"



Scene*  BIA_msp_Rotate3(Scene *scn, 
		double thx, double thy, double thz, // angles to rotate
		  int cx, int cy, int cz, // center of the rotation
                double transx, double transy, double transz) // translation after rotation
{
  int p;
  Voxel u;
  Scene *res;
  RealMatrix *trans1,*rot1,*rot2,*rot3,*trans2,*aux1,*aux2,*inv,*vox;
  trans1 = TranslationMatrix3(-cx,-cy,-cz);
  rot1 = RotationMatrix3(0,thx);
  rot2 = RotationMatrix3(1,thy);
  rot3 = RotationMatrix3(2,thz);
  trans2 = TranslationMatrix3(cx+transx,cy+transy,cz+transz);
  // Compose transform
  aux1 = MultRealMatrix(trans2,rot3);
  aux2 = MultRealMatrix(aux1,rot2);
  DestroyRealMatrix(&aux1);
  aux1 = MultRealMatrix(aux2,rot1);
  DestroyRealMatrix(&aux2);
  aux2 = MultRealMatrix(aux1,trans1);
  inv = InvertRealMatrix(aux2);
  DestroyRealMatrix(&trans1);
  DestroyRealMatrix(&rot1);
  DestroyRealMatrix(&rot2);
  DestroyRealMatrix(&rot3);
  DestroyRealMatrix(&trans2);
  DestroyRealMatrix(&aux2);
  DestroyRealMatrix(&aux1);
  // Applying transform for all voxels
  res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < res->zsize; u.z++) 
    for (u.y=0; u.y < res->ysize; u.y++)
      for (u.x=0; u.x < res->xsize; u.x++){
	p = u.x + res->tby[u.y] + res->tbz[u.z];
	vox = TransformVoxel(inv,u); // acelerar isso!!!!
	if ((vox->val[0][0]<=res->xsize-1)&&(vox->val[0][1]<=res->ysize-1)
	    &&(vox->val[0][2]<=res->zsize-1 && vox->val[0][0]>0 && vox->val[0][1]>0 && vox->val[0][2]>0))
	  {
	    res->data[p]=GetVoxelValue_trilinear(scn,vox->val[0][0],vox->val[0][1],vox->val[0][2]);
	  }
	else
	  res->data[p]=0;
	DestroyRealMatrix(&vox);
      }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}



/*

Scene*  BIA_msp_Rotate3_Bin(Scene *scn, 
		double thx, double thy, double thz, // angles to rotate
		  int cx, int cy, int cz, // center of the rotation
                double transx, double transy, double transz) // translation after rotation
{
  int p;
  Voxel u;
  Scene *res;
  RealMatrix *trans1,*rot1,*rot2,*rot3,*trans2,*aux1,*aux2,*inv,*vox;
  trans1 = TranslationMatrix3(-cx,-cy,-cz);
  rot1 = RotationMatrix3(0,thx);
  rot2 = RotationMatrix3(1,thy);
  rot3 = RotationMatrix3(2,thz);
  trans2 = TranslationMatrix3(cx+transx,cy+transy,cz+transz);
  // Compose transform
  aux1 = MultRealMatrix(trans2,rot3);
  aux2 = MultRealMatrix(aux1,rot2);
  DestroyRealMatrix(&aux1);
  aux1 = MultRealMatrix(aux2,rot1);
  DestroyRealMatrix(&aux2);
  aux2 = MultRealMatrix(aux1,trans1);
  inv = InvertRealMatrix(aux2);
  DestroyRealMatrix(&trans1);
  DestroyRealMatrix(&rot1);
  DestroyRealMatrix(&rot2);
  DestroyRealMatrix(&rot3);
  DestroyRealMatrix(&trans2);
  DestroyRealMatrix(&aux2);
  DestroyRealMatrix(&aux1);
  // Applying transform for all voxels
  float val;
  res = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < res->zsize; u.z++) 
    for (u.y=0; u.y < res->ysize; u.y++)
      for (u.x=0; u.x < res->xsize; u.x++){
	p = u.x + res->tby[u.y] + res->tbz[u.z];
	vox = TransformVoxel(inv,u); // acelerar isso!!!!
	if ((vox->val[0][0]<=res->xsize-1)&&(vox->val[0][1]<=res->ysize-1)
	    &&(vox->val[0][2]<=res->zsize-1 && vox->val[0][0]>0 && vox->val[0][1]>0 && vox->val[0][2]>0))
	  {
	    val = GetVoxelValue_trilinear(scn,vox->val[0][0],vox->val[0][1],vox->val[0][2]);
	    if (val<0.5) res->data[p]=0;
	    else res->data[p]=1;
	  }
	else
	  res->data[p]=0;
	DestroyRealMatrix(&vox);
      }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}

*/



Scene* BIA_msp_RotateAxis(Scene *scn, int plane)  // plane= 1(X) 2(Y) 3(Z) -1(-X) -2(-Y) -3(-Z)
{
  Scene *res=NULL;
  int i,j,k,p,q;

  if (plane==1) {
      res = CreateScene(scn->xsize,scn->zsize,scn->ysize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[j] + res->tby[res->ysize-k-1] + i; // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==2) {
      res = CreateScene(scn->zsize,scn->ysize,scn->xsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[res->zsize-i-1] + res->tby[j] + k;  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==3)          {
      res = CreateScene(scn->ysize,scn->xsize,scn->zsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++)     {
              p = res->tbz[k] + res->tby[i] + (res->xsize-j-1);  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==-1) {
      res = CreateScene(scn->xsize,scn->zsize,scn->ysize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[res->zsize-j-1] + res->tby[k] + i; // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==-2) {
      res = CreateScene(scn->zsize,scn->ysize,scn->xsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++) {
              p = res->tbz[i] + res->tby[j] + (res->xsize-k-1);  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  if (plane==-3)          {
      res = CreateScene(scn->ysize,scn->xsize,scn->zsize);
      for (i=0;i<scn->xsize;i++)
        for (j=0;j<scn->ysize;j++)
          for (k=0;k<scn->zsize;k++)     {
              p = res->tbz[k] + res->tby[res->ysize-i-1] + j;  // res voxel
              q = scn->tbz[k] + scn->tby[j] + i;  // scn voxel
              res->data[p]=scn->data[q];
            }
    }
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}



Scene*  BIA_msp_RotateToMSP(Scene *scn, Plane *p)
{
    Point center;
    Scene *res;
    float thx,thy,transz;
    float A,B,C,D;
    A = p->normal.x;
    B = p->normal.y;
    C = p->normal.z;
    D = -(p->normal.x*p->po.x + p->normal.y*p->po.y + p->normal.z*p->po.z);
    center.x = scn->xsize/2.0;
    center.y = scn->ysize/2.0;
    center.z = (-D-B*center.y-A*center.x)/C;
    transz = ((int)scn->zsize/2) - center.z;  // put the MSP on the zsize/2 slice
    if (C==0) {printf("error: thx=0\n"); exit(1);}
       else thx = atan(B/C);
    if (C==0) {printf("error: thx=0\n"); exit(1);}
    else {
      thy = - atan(A/(B*sin(thx)+C*cos(thx)));
    }
    res = BIA_msp_Rotate3(scn,thx,thy,0, center.x,center.y,center.z, 0,0,transz);
    return res;
}





Scene* BIA_msp_RotateToMSP_withPermute(Scene *in, Plane *msp, int input_ori, int detected_plane)
{
  int verbose=1;
  Scene *scn,*out;
  scn = in;
  // Reslicing the scene to sagittal orientation
  if (input_ori==2) { // axial
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(scn,1);
    scn  = BIA_msp_RotateAxis(tmp,2);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==3) { // coronal
    if (verbose) { printf("Reslicing to sagittal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(scn,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }

  if (input_ori==0 && detected_plane==1) {  // axial or coronal
    if (verbose) { printf("Reslicing to sagittal...");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(scn,2);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==0 && detected_plane==2) { // laid-down coronal
    if (verbose) { printf("Reslicing to sagittal...");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(scn,1);
    scn = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  

  // Rotate the volume according to MSP found
  if (verbose) { printf("Rotating... ");  fflush(stdout); }
  out = BIA_msp_RotateToMSP(scn,msp);
  if (verbose) { printf("done.\n");  fflush(stdout); }

  // Reslicing the output
  if (input_ori==2) { // axial
    if (verbose) { printf("Reslicing back to axial... ");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out  = BIA_msp_RotateAxis(tmp,-1);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==3) { // coronal
    if (verbose) { printf("Reslicing back to coronal... ");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }

  if (input_ori==0 && detected_plane==1) {  // axial or coronal
    if (verbose) { printf("Reslicing back to X=Xsize/2...");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(out,-2);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  if (input_ori==0 && detected_plane==2) { // laid-down coronal
    if (verbose) { printf("Reslicing back to Y=Ysize/2...");  fflush(stdout); }
    Scene *tmp;
    tmp = BIA_msp_RotateAxis(out,-1);
    DestroyScene(&out);
    out = CopyScene(tmp);
    DestroyScene(&tmp);
    if (verbose) { printf("done.\n");  fflush(stdout); }
  }
  
  return out;
}










// Parameters:
// scn: Input scene
// mask: consider just the voxels within this mask (0=bg / 1=fg) or NULL
// input_ori: 1=sagittal / 2=axial / 3=coronal / 0=unknown
// quality: 1=best accuracy/slowest / 2=moderate / 3=medium accuracy/fastest
Scene* BIA_MSP_Align(Scene *in, Scene *mask, int input_ori, int quality, ArrayList *segmobjs, Plane **msp)
{
  //int verbose=1;
  int aligned=0;

  Scene *out;

  if (input_ori<1 || input_ori>3) input_ori=0; //default=0
  //if (output_ori<1 || output_ori>3) output_ori=1; //default=1  

  int detected_plane = 0;

  if( *msp != NULL )
    aligned = 1;
  // Compute the MSP
  if( aligned == 0 ) {
    *msp=FindSymmetryPlane(in,mask,input_ori,quality,&detected_plane);
    // Apply the msp
    out = BIA_msp_RotateToMSP_withPermute(in, *msp, input_ori, detected_plane);
  }
  else
    out = BIA_msp_RotateToMSP_withPermute(in, *msp, 2, 1);

  SegmObject *obj=NULL;
  int i;
  int n = segmobjs->n;
  for (i=0; i < n; i++) {
    obj = (SegmObject *)GetArrayListElement(segmobjs, i);
    Scene *mask1 = CreateScene(in->xsize,in->ysize,in->zsize);
    CopyBMap2SceneMask(mask1,obj->mask);
    Scene *mask2;
    if( aligned == 0 )
      mask2 = BIA_msp_RotateToMSP_withPermute(mask1, *msp, input_ori, detected_plane);
    else
      mask2 = BIA_msp_RotateToMSP_withPermute(mask1, *msp, 2, 1);
    DestroyScene(&mask1);
    BMap *newbmap;
    newbmap = SceneMask2BMap(mask2);
    DestroyScene(&mask2);
    BMapCopy(obj->mask,newbmap);
    BMapDestroy(newbmap);
  }
  return out;
}



