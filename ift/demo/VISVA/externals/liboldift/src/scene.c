//#include <endian.h>
#include <ctype.h>
#include "scene.h"
#include "adjacency.h"
#include "common.h"
#include "curve.h"
#include "geometry.h"
#include "analysis.h"
#include "radiometric3.h"
#include "segmentation3.h"
#include "mathematics3.h"
#include "realmatrix.h"
#include "file_analyze.h"
#include "compressed.h"
#include "nifti1_io.h"

void CopyVoxelSize(Scene *scn1, Scene *scn2)
{
  scn2->dx = scn1->dx;
  scn2->dy = scn1->dy;
  scn2->dz = scn1->dz;
}


Scene  *CreateScene(int xsize,int ysize,int zsize)
{
  Scene *scn=NULL;
  int i,xysize;

  scn = (Scene *) calloc(1,sizeof(Scene));
  if (scn == NULL){
    Error(MSG1,"CreateScene");
  }
  
  scn->data    = AllocIntArray(xsize*ysize*zsize);
  scn->xsize   = xsize;
  scn->ysize   = ysize;
  scn->zsize   = zsize;
  scn->dx      = 1.0;
  scn->dy      = 1.0;
  scn->dz      = 1.0;
  scn->tby     = AllocIntArray(ysize);
  scn->tbz     = AllocIntArray(zsize);
  scn->maxval  = 0;
  scn->n       = xsize*ysize*zsize;
  scn->nii_hdr = NULL;

  if (scn->data==NULL || scn->tbz==NULL || scn->tby==NULL) {
    Error(MSG1,"CreateScene");
  }

  scn->tby[0]=0;
  for (i=1; i < ysize; i++)
    scn->tby[i]=scn->tby[i-1] + xsize;

  scn->tbz[0]=0; xysize = xsize*ysize;
  for (i=1; i < zsize; i++)
    scn->tbz[i]=scn->tbz[i-1] + xysize;

  return(scn);
}

Scene *CopyScene(Scene *scn){
  Scene *aux = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  aux->dx     = scn->dx;
  aux->dy     = scn->dy;
  aux->dz     = scn->dz;
  aux->maxval = scn->maxval;
  aux->nii_hdr = NULL;

  if( scn->nii_hdr != NULL ) {
    aux->nii_hdr = nifti_copy_nim_info( scn->nii_hdr );
  }
  aux->n=scn->xsize*scn->ysize*scn->zsize;
  memcpy(aux->data,scn->data,sizeof(int) * aux->n);
  return aux;
}

void     DestroyScene(Scene **scn)
{
  Scene *aux;

  aux = *scn;
  if(aux != NULL){
    if (aux->data != NULL)  free(aux->data); 
    if (aux->tby  != NULL)  free(aux->tby);
    if (aux->tbz  != NULL)  free(aux->tbz);
    if (aux->nii_hdr != NULL) nifti_image_free( aux->nii_hdr );
    free(aux);    
    *scn = NULL;
  }
}

bool ValidVoxel(Scene *scn, int vx, int vy, int vz)
{
  if ((vx >= 0)&&(vx < scn->xsize)&&
      (vy >= 0)&&(vy < scn->ysize)&&
      (vz >= 0)&&(vz < scn->zsize))
    return(true);
  else
    return(false);
}
Voxel   Transform_Voxel(float M[4][4],Voxel v){
  Voxel nv;

  nv.x = (int) (M[0][0]*v.x + M[0][1]*v.y + M[0][2]*v.z + M[0][3]);
  nv.y = (int) (M[1][0]*v.x + M[1][1]*v.y + M[1][2]*v.z + M[1][3]);
  nv.z = (int) (M[2][0]*v.x + M[2][1]*v.y + M[2][2]*v.z + M[2][3]);

  return(nv);

}

int MaximumValue3(Scene *scn)
{
  unsigned int i, n, r;
  int Imax;
      
  Imax = 0;
  n = scn->xsize*scn->ysize*scn->zsize - 1;
  r = n%4;
  n -= r;
  for (i=0; i < n; i+=4) {
    if (scn->data[i] > Imax)
      Imax = scn->data[i];
    if (scn->data[i+1] > Imax)
      Imax = scn->data[i+1];
    if (scn->data[i+2] > Imax)
      Imax = scn->data[i+2];
    if (scn->data[i+3] > Imax)
      Imax = scn->data[i+3];
  }
  while (r != 0) {
    if (scn->data[i+r-1] > Imax)
      Imax = scn->data[i+r-1];
    --r;
  }

  scn->maxval = Imax;

  return(Imax); 
}

void SetSceneImax(Scene *scn, int val) {
  scn->maxval = val;
}

int SceneImax(Scene *scn)
{
  return scn->maxval;
}

int MinimumValue3(Scene *scn)
{
  int i,n;
  int Imin;
  
  Imin = 65535; 
  n = scn->xsize*scn->ysize*scn->zsize;
  for (i=0; i < n; i++) 
    if (scn->data[i] < Imin)
      Imin = scn->data[i];
  
  return(Imin); 
}


int   MaximumValueMask3(Scene *scn, Scene *mask){
  int n,p,max;

  max = INT_MIN;
  n = scn->xsize*scn->ysize*scn->zsize;
  for(p=0; p<n; p++){
    if(mask->data[p]>0)
      if(scn->data[p]>max)
	max = scn->data[p];
  }
  return max;
}


int   MinimumValueMask3(Scene *scn, Scene *mask){
  int n,p,min;

  min = INT_MAX;
  n = scn->xsize*scn->ysize*scn->zsize;
  for(p=0; p<n; p++){
    if(mask->data[p]>0)
      if(scn->data[p]<min)
	min = scn->data[p];
  }
  return min;
}


Scene *ROI3(Scene *scn, int xl, int yl, int zl, int xh, int yh, int zh)
{
  Scene *roi=NULL;
  Voxel v;
  int i,j;
  
  if (ValidVoxel(scn,xl,yl,zl)&&ValidVoxel(scn,xh,yh,zh)&&
      (xl <= xh)&&(yl<=yh)&&(zl<=zh)){
    roi=CreateScene(xh-xl+1,yh-yl+1,zh-zl+1);
    roi->dx = scn->dx;
    roi->dy = scn->dy;
    roi->dz = scn->dz;
    j = 0;
    for (v.z=zl; v.z <= zh; v.z++)
      for (v.y=yl; v.y <= yh; v.y++)
	for (v.x=xl; v.x <= xh; v.x++) {
	  i = v.x + scn->tby[v.y] + scn->tbz[v.z];
	  roi->data[j] = scn->data[i];
	  j++;
	}
  }

  return(roi);
}

Scene *MBB3(Scene *scn) 
{
  Voxel v,lower,higher;
  Scene *mbb=NULL;
  
  lower.x  = scn->xsize-1;
  lower.y  = scn->ysize-1;
  lower.z  = scn->zsize-1;
  higher.x = 0;
  higher.y = 0;
  higher.z = 0;
  
  for (v.z=0; v.z < scn->zsize; v.z++)
    for (v.y=0; v.y < scn->ysize; v.y++)
      for (v.x=0; v.x < scn->xsize; v.x++)    
	if (scn->data[v.x+scn->tby[v.y]+scn->tbz[v.z]] > 0)
	  {
	    if (v.x < lower.x)
	      lower.x = v.x;
	    if (v.y < lower.y)
	      lower.y = v.y;
	    if (v.z < lower.z)
	      lower.z = v.z;
	    if (v.x > higher.x)
	      higher.x = v.x;
	    if (v.y > higher.y)
	      higher.y = v.y;	
	    if (v.z > higher.z)
	      higher.z = v.z;	
	  }  
  mbb = ROI3(scn,lower.x,lower.y,lower.z,higher.x,higher.y,higher.z);

  return(mbb);	
}

#ifdef _MSC_VER
#define strcasecmp stricmp
#endif

Scene *ReadScene(char *filename)
{
  Scene  *scn=NULL;	
  FILE   *fp=NULL;
  uchar  *data8=NULL;
#if _WIN32 || __BYTE_ORDER==__LITTLE_ENDIAN 
  ushort *data16=NULL;
#endif
  char    type[10];
  int     i,n,v,xsize,ysize,zsize;
  long    pos;
  int     nread;

  // Checking file type
  int len = strlen(filename);
  if ( (len>=4) && ((strcasecmp(filename + len - 4, ".hdr")==0) || (strcasecmp(filename + len - 4, ".img")==0))) 
    /*return ReadScene_Analyze(filename); */
    return ReadScene_Nifti1(filename);
  if ( (len>=8) && (strcasecmp(filename + len - 8, ".scn.bz2")==0))
    return ReadCompressedScene(filename);
  if ( (len>=4) && (strcasecmp(filename + len - 4, ".nii")==0))
    return ReadScene_Nifti1(filename);
  if ( (len>=7) && (strcasecmp(filename + len - 7, ".nii.gz")==0))
    return ReadScene_Nifti1(filename);
  if ( (len<=4) || (strcasecmp(filename + len - 4, ".scn")!=0)) {
    Error(MSG2,"ReadScene: Invalid file name or extension.");
    return NULL;
  }

  // Read the scn file
  fp = fopen(filename,"rb");
  if (fp == NULL){
    Error(MSG2,"ReadScene");
  } 
  if((nread = fscanf(fp,"%s\n",type)) != 1)
    Error("Error when reading the scene image type!", "ReadScene");

  if((strcmp(type,"SCN")==0)){
    if((nread = fscanf(fp,"%d %d %d\n",&xsize,&ysize,&zsize)) != 3)
       Error("Error when reading the scene image size!", "ReadScene");
    scn = CreateScene(xsize,ysize,zsize);
    n = xsize*ysize*zsize;
    if((nread = fscanf(fp,"%f %f %f\n",&scn->dx,&scn->dy,&scn->dz)) != 3)
      Error("Error when reading the scene image pixel resolution", "ReadScene");
    if((nread = fscanf(fp,"%d",&v)) != 1)
      Error("Error when reading the number of bits per pixel of the scene!", "ReadScene");

    pos = ftell(fp);
    //printf(" current relative position in file: %ld\n",pos);///	
    fseek(fp,(pos+1)*sizeof(char),SEEK_SET); // +1 for the EOL \n character not included in last fscanf() call
    if (v==8){			
      data8  = AllocUCharArray(n);			
      if((nread = fread(data8,sizeof(uchar),n,fp)) != n)
	Error("Error when reading the 8 bit data of the scene!", "ReadScene");
      for (i=0; i < n; i++) 
	scn->data[i] = (int) data8[i];
      free(data8);
    } else if (v==16) {
      // assuming that data was written with LITTLE-ENDIAN Byte ordering, i.e. LSB...MSB
#if _WIN32 || __BYTE_ORDER==__LITTLE_ENDIAN
      // for PCs, Intel microprocessors (LITTLE-ENDIAN too) -> no change
      data16 = AllocUShortArray(n);
      if((nread = fread(data16,sizeof(ushort),n,fp)) != n)
	Error("Error when reading the 16 bit data of the scene!", "ReadScene");;
      for (i=0; i < n; i++)
	scn->data[i] = (int) data16[i];
      free(data16);
#else
      // for Motorola, IBM, SUN (BIG-ENDIAN) -> SWAp Bytes!
      Warning("Data is converted from LITTLE to BIG-ENDIAN","ReadScene");
      data8 = AllocUCharArray(2*n);
      if((nread = fread(data8,sizeof(uchar),2*n,fp)) != 2*n)
	Error("Error when reading the 8 bit data of the scene!", "ReadScene");
      j=0;
      for (i=0; i < 2*n; i+=2) {
	scn->data[j] = (int) data8[i] + 256 * (int) data8[i+1];
	j++;
      }
      free(data8);
#endif
      /*
      // assuming that data was written with BIG-ENDIAN Byte ordering, i.e. MSB...LSB
      #ifdef _WIN32
      // for PCs, Intel microprocessors (LITTLE-ENDIAN) -> SWAp Bytes!
      Warning("Data is converted from BIG to LITTLE-ENDIAN","ReadScene");
      data8 = AllocUCharArray(2*n);
      if((nread = fread(data8,sizeof(uchar),2*n,fp);
      j=0;
      for (i=0; i < 2*n; i+=2) {
      scn->data[i] = (int) data8[i] * 256 + (int) data8[i+1];
      j++;
      }
      free(data8);
      #else
      // for Motorola, IBM, SUN (BIG-ENDIAN too) -> no change
      data16 = AllocUShortArray(n);
      if((nread = fread(data16,sizeof(ushort),n,fp);
      for (i=0; i < n; i++)
      scn->data[i] = (int) data16[i];
      free(data16);
      #endif
      */
      
    } else { /* n = 32 */
      //Warning("32-bit data. Values may be wrong (little/big endian byte ordering)","ReadScene");
      n = xsize*ysize*zsize;
      if((nread = fread(scn->data,sizeof(int),n,fp)) != n)
	Error("Error when reading the 32 bit data of the scene!", "ReadScene");      
    }
    fclose(fp);
  } else {
    fprintf(stderr,"Input scene must be SCN\n");
    exit(-1);
  }
  scn->maxval = MaximumValue3(scn);
  return(scn);
}

void WriteScene(Scene *scn, char *filename) 
{
  FILE *fp=NULL;
  int Imax;
  int i,n;
  uchar  *data8 =NULL;
  ushort *data16=NULL;

  // Checking file type
  int len = strlen(filename);
  if ( (len>=4) && ((strcasecmp(filename + len - 4, ".hdr")==0) || (strcasecmp(filename + len - 4, ".img")==0))) {
    //WriteScene_Analyze(scn, filename);
    WriteScene_Nifti1(scn, filename);
    return;
  }
  if ( (len>=8) && (strcasecmp(filename + len - 8, ".scn.bz2")==0)) {
    WriteCompressedScene(scn, filename);
    return;
  }
  if ( (len>=7) && (strcasecmp(filename + len - 7, ".nii.gz")==0)) {
    WriteScene_Nifti1( scn, filename );
    return;
  }
  if ( (len>=4) && (strcasecmp(filename + len - 4, ".nii")==0)) {
    WriteScene_Nifti1( scn, filename );
    return;
  }
  if ( (len<=4) || (strcasecmp(filename + len - 4, ".scn")!=0)) {
    Error(MSG2,"WriteScene: Invalid file name or extension.");
  }



  // Writing the scn file
  fp = fopen(filename,"wb"); 
  if (fp == NULL) 
    Error(MSG2,"WriteScene");

  fprintf(fp,"SCN\n");
  fprintf(fp,"%d %d %d\n",scn->xsize,scn->ysize,scn->zsize);
  fprintf(fp,"%f %f %f\n",scn->dx,scn->dy,scn->dz);
  
  Imax = MaximumValue3(scn);
  
  n = scn->xsize*scn->ysize*scn->zsize;
  if (Imax < 256) {
    fprintf(fp,"%d\n",8);
    data8 = AllocUCharArray(n);
    for (i=0; i < n; i++) 
      data8[i] = (uchar) scn->data[i];
    fwrite(data8,sizeof(uchar),n,fp);
    free(data8);
  } else if (Imax < 65536) {
    fprintf(fp,"%d\n",16);
    data16 = AllocUShortArray(n);
    for (i=0; i < n; i++)
      data16[i] = (ushort) scn->data[i];
    fwrite(data16,sizeof(ushort),n,fp);
    free(data16);
  } else {
    fprintf(fp,"%d\n",32);
    fwrite(scn->data,sizeof(int),n,fp);
  }
  fclose(fp);
}

Image *GetSlice(Scene *scn, int z)
{
  Image *img=NULL;
  int n; /* , Imax, s, n1; */
  int *data; // i;
  
  img  = CreateImage(scn->xsize,scn->ysize);
  n    = img->ncols*img->nrows;              // n = number of pixels 
  data = scn->data + z*n;
  n    = n*sizeof(int);                        // n = number of bytes 
  memcpy(img->val, data, n);                 // copy n bytes 
  
  /*
  img = CreateImage(scn->xsize, scn->ysize);
  n = img->ncols*img->nrows;
  data = scn->data + z*n;

  for(i = 0; i < n; i++)
    img->val[i] = data[i];
  */
  return(img);
}

void PutSlice(Image *img, Scene *scn, int z)
{
  int n; //i;
  int *data;

  n    = img->ncols*img->nrows;            // n = number of pixels 
  data = scn->data + z*n;
  n    = n*sizeof(int);                        // n = number of bytes 
  memcpy(data, img->val, n);                 // copy n bytes 
  
  /*
  n = img->ncols*img->nrows;
  data = scn->data + z*n;

  for(i = 0; i < n; i++)
    data[i] = img->val[i];
  */
}

Image *GetXSlice(Scene *scn, int x)
{
  Image *img=NULL;
  int i,j;
  Voxel v;

  v.x = x;
  img = CreateImage(scn->ysize,scn->zsize);
  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.y=0;v.y<scn->ysize;v.y++) {
      i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
      if (scn->data[i] != 0) {
	j = img->tbrow[v.z] + v.y;
	img->val[j] = scn->data[i];
      }
    }
  return(img);
}


void PutXSlice(Image *img, Scene *scn, int x)
{
  int i,j;
  Voxel v;

  v.x = x;
  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.y=0;v.y<scn->ysize;v.y++) {
      i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
      j = img->tbrow[v.z] + v.y;
      scn->data[i]= img->val[j];
    }
}


Image *GetYSlice(Scene *scn, int y)
{
  Image *img=NULL;
  int i,j;
  Voxel v;

  v.y = y;
  img = CreateImage(scn->xsize,scn->zsize);
  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.x=0;v.x<scn->xsize;v.x++) {
      i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
      if (scn->data[i] != 0) {
	j = img->tbrow[v.z] + v.x;
	img->val[j] = scn->data[i];
      }
    }
  return(img);
}

void PutYSlice(Image *img, Scene *scn, int y)
{
  int i,j;
  Voxel v;

  v.y = y;
  for (v.z=0;v.z<scn->zsize;v.z++)
    for (v.x=0;v.x<scn->xsize;v.x++) {
      i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
      j = img->tbrow[v.z] + v.x;
      scn->data[i]= img->val[j];
    }
}

int VoxelValue(Scene *scn,Voxel v){
  int k,aux;
  if(ValidVoxel(scn,v.x,v.y,v.z)){
  k=v.x+scn->tby[v.y]+scn->tbz[v.z];
  aux=scn->data[k];
  } else aux=0;
  
  return aux;
}

Scene *GetObject(Scene *scn,int obj) {

  int i,n;
  Scene *objscn;

  n = scn->xsize*scn->ysize*scn->zsize;
  objscn= CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (i=0; i < n; i++) 
    if (scn->data[i] == obj)
      objscn->data[i] = obj;
  
  return(objscn); 

}

Scene *LinearInterp(Scene *scn,float dx,float dy,float dz){
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
    tmp = CreateScene(ROUND((float)(scene->xsize)*scene->dx/dx),scene->ysize,scene->zsize);
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

            value = ROUND((( scn->dx - dist_PQ)*(float)VoxelValue(scene,P) + dist_PQ * (float)VoxelValue(scene,R) )/scn->dx);
            tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
	  }
    scene=tmp;
  }

  /* Resample in y */

  if (dy != scn->dy) {
    tmp = CreateScene(scene->xsize, ROUND((float)scene->ysize * scn->dy / dy),scene->zsize);
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
            value = ROUND((( (scn->dy - dist_PQ)*(float)VoxelValue(scene,P)) + dist_PQ * (float)VoxelValue(scene,R)) / scn->dy) ;
	    tmp->data[Q.x+tmp->tby[Q.y]+tmp->tbz[Q.z]]=value;
           
          }
    if (scene != scn) {
      DestroyScene(&scene);
    }
    scene=tmp;
  }

  /* Resample in z */

  if (dz != scn->dz) {
    tmp = CreateScene(scene->xsize,scene->ysize,ROUND((float)scene->zsize * scn->dz / dz));
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

	    value = ROUND((( (scn->dz - dist_PQ)*(float)VoxelValue(scene,P)) + dist_PQ * (float)VoxelValue(scene,R)) / scn->dz) ;
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

Scene *KnnInterp(Scene *scn,float dx,float dy,float dz){
  Scene *new_scn;
  float min, x, y, z;
  Voxel u;

  /* The voxel sizes default in the input scene should be
     dx=dy=dz=1.0 */

  if ((scn->dx == 0.0) && (scn->dy == 0.0) && (scn->dz == 0.0)) {    
    scn->dx=1.0;
    scn->dy=1.0;
    scn->dz=1.0;
  }

  /* The voxel sizes default in the output scene should be
     dx=dy=dz=min(dx,dy,dz) */

  if ((dx == 0.0) || (dy == 0.0) || (dz == 0.0)) {
    min = FLT_MAX;
    if (scn->dx < min)
      min = scn->dx;
    if (scn->dy < min)
      min = scn->dy;
    if (scn->dz < min)
      min = scn->dz;
    dx = min; dy = min; dz = min;
    if (min <= 0.0) {
      fprintf(stderr,"Voxel distance cannot be negative.\n");
      exit(-1);
    }
  }

  /* If there is no need for resampling then return the input scene */

  if ((dx == scn->dx) && (dy == scn->dy) && (dz == scn->dz)) {
    new_scn = CopyScene(scn);
  } else {
    /* do knn interpolation */
    new_scn = CreateScene(ROUND((float)(scn->xsize*scn->dx/dx)),ROUND((float)(scn->ysize*scn->dy/dy)),ROUND((float)(scn->zsize-1)*scn->dz/dz));
    new_scn->dx     = dx;
    new_scn->dy     = dy;
    new_scn->dz     = dz;
    dx              = dx / scn->dx;
    dy              = dy / scn->dy;
    dz              = dz / scn->dz;
    new_scn->maxval = scn->maxval;
    for (z=0.0,u.z=0; z < scn->zsize; z=z+dz, u.z++)
      for (y=0.0, u.y=0; y < scn->ysize; y=y+dy, u.y++)
	for (x=0.0, u.x=0; x < scn->xsize; x=x+dx, u.x++){
	  
	  if ((x==(int)x)&&(y==(int)y)&&(z==(int)z)){
	    new_scn->data[u.x + new_scn->tby[u.y] + new_scn->tbz[u.z]]=
	      scn->data[(int)x + scn->tby[(int)y] + scn->tbz[(int)z]];
	  }else{
	    new_scn->data[u.x + new_scn->tby[u.y] + new_scn->tbz[u.z]]=KnnInterpValue(scn, x, y, z); 	    
	  }
	}
  }
  return(new_scn);
}

#define KNNSIZE 7

int KnnInterpValue(Scene *scn, float x, float y, float z)
{
  int j;
  Voxel u, v;
  Voxel vox[KNNSIZE];
  int value=scn->data[(int)x+scn->tby[(int)y]+scn->tbz[(int)z]];
  int absdif[KNNSIZE];
  int d;
  int i,q;
  float new_value, w, cw;

  for (i=0; i < KNNSIZE; i++) 
    absdif[i]=INT_MAX;

  j=0;
  for (u.z = (int)z-1; u.z <= (int)z+1; u.z++) 
    for (u.y = (int)y-1; u.y <= (int)y+1; u.y++) 
      for (u.x = (int)x-1; u.x <= (int)x+1; u.x++) {

	if (ValidVoxel(scn,u.x,u.y,u.z)){
	  j++;
	  q           = u.x+scn->tby[u.y]+scn->tbz[u.z];
	  i           = KNNSIZE-1;
	  absdif[i]   = abs(scn->data[q]-value);
	  vox[i]    = u;

	  // find the nearest neighbors
	  while ((i > 0)&&(absdif[i]<absdif[i-1])) {
	    d           = absdif[i];
	    v           = vox[i];
	    absdif[i]   = absdif[i-1];
	    vox[i]    = vox[i-1];
	    absdif[i-1] = d;
	    vox[i-1]  = v;
	    i--;
	  }
	}

      }

  if (j < KNNSIZE) return((int)value);

  // compute interpolation
  new_value = 0.0; cw = 0.0;
  for (i=0; i < KNNSIZE-1; i++) {
    // w = (vox[i].x - x)*(vox[i].x - x) + 
    //  (vox[i].y - y)*(vox[i].y - y) + 
    //  (vox[i].z - z)*(vox[i].z - z);
    //  w = exp(-w/4.0);
    //  w = 1.0/(absdif[i]+1.0);
    w = KNNSIZE-i;
    cw += w;
    new_value += scn->data[vox[i].x+scn->tby[vox[i].y] + scn->tbz[vox[i].z]]*w;
  }

  return((int)(new_value/cw));
}
 	    
Scene *ImageSurface(Image *img)
{
  Scene *scn=NULL;
  int x,y,z;

  scn = CreateScene(img->ncols,img->nrows,MaximumValue(img)+1);
  
  for (y=0; y < img->nrows; y++) 
    for (x=0; x < img->ncols; x++) 
      for (z=0; z <= img->val[x+img->tbrow[y]]; z++)
	scn->data[x+scn->tby[y]+scn->tbz[z]]=1;
  
  return(scn);
}


bool EdgeVoxel(Scene *scn, int v, int s) {
  int x, y;
  /*  z = v / s; */
  y = (v % s) / scn->xsize;
  x = (v % s) % scn->xsize;

  return ((x == 0) || (y == 0) || 
	  (x == scn->xsize-1) || (y == scn->ysize - 1));
}


int GetNSlices(Scene *scn) {
  return scn->zsize;
}

int GetXSize(Scene *scn) {
  return scn->xsize;
}

int GetYSize(Scene *scn) {
  return scn->ysize;
}

int GetZSize(Scene *scn) {
  return scn->zsize;
}

float GetDx(Scene *scn) {
  return scn->dx;
}

float GetDy(Scene *scn) {
  return scn->dy;
}

float GetDz(Scene *scn) {
  return scn->dz;
}

void SetDx(Scene *scn, float dx) {
  scn->dx = dx;
}

void SetDy(Scene *scn, float dy) {
  scn->dy = dy;
}

void SetDz(Scene *scn, float dz) {
  scn->dz = dz;
}

void SetVoxelSize(Scene *scn, float dx, float dy, float dz){
  SetDx(scn, dx);
  SetDy(scn, dy);
  SetDz(scn, dz);
}

int Coord2Voxel(Scene *scn, int x, int y, int z) {
  return (x + scn->tby[y] + scn->tbz[z]);
}

int GetVoxelValue(Scene *scn, int voxel) {
  return (scn->data[voxel]);
}

void SetScene(Scene *scn, int value)
{ 
  int i,n;
  n = scn->xsize*scn->ysize*scn->zsize;
  for (i=0; i < n; i++){
    scn->data[i]=value;
  }
  scn->maxval = value;
}

void SetLabelScene(Scene *scn, int label, int value)
{ 
  int i,n;
  n = scn->xsize*scn->ysize*scn->zsize;
  for (i=0; i < n; i++){
    if (scn->data[i] == label)
      scn->data[i]=value;
  }
  scn->maxval = MaximumValue3(scn);
}

Scene *AddFrame3(Scene *scn, int sz, int value)
{
  Scene *fscn;
  int y, z,*dst,*src,nbytes,offset1, offset2;
  
  fscn = CreateScene(scn->xsize+(2*sz),scn->ysize+(2*sz), scn->zsize+(2*sz));
  fscn->dx = scn->dx;
  fscn->dy = scn->dy;
  fscn->dz = scn->dz;

  SetScene(fscn,value);
  nbytes = sizeof(int)*scn->xsize;
  
  offset1 = 0;
  offset2 = sz+ fscn->tby[sz] + fscn->tbz[sz];
  
  for(z=0; z<scn->zsize; z++){
    src = scn->data+offset1;
    dst = fscn->data+offset2;
    for (y=0; y<scn->ysize; y++){
      memcpy(dst,src,nbytes);
      src += scn->xsize;
      dst += fscn->xsize;
    }
    offset1 += scn->xsize*scn->ysize;
    offset2 += fscn->xsize*fscn->ysize;
  }
  
  return(fscn);
}

Scene *RemFrame3(Scene *fscn, int sz) {
  Scene *scn;
  int y,z,*dst,*src,nbytes,offset;

  scn = CreateScene(fscn->xsize-(2*sz),fscn->ysize-(2*sz),fscn->zsize-(2*sz));
  scn->dx = fscn->dx;
  scn->dy = fscn->dy;
  scn->dz = fscn->dz;

  nbytes = sizeof(int)*scn->xsize;  
  offset = sz + fscn->tby[sz] + fscn->tbz[sz];
  src = fscn->data+offset;
  dst = scn->data;
  for (z=0; z < scn->zsize; z++,src+=2*sz*fscn->xsize) {
    for (y=0; y < scn->ysize; y++,src+=fscn->xsize,dst+=scn->xsize){
      memcpy(dst,src,nbytes);
    }
  }
  return(scn);
}


Scene *DrawBorder3(Scene *scn, Scene *label, int value)
{
  Scene *hscn=NULL;
  int p,q,i;
  AdjRel3 *A=NULL;
  Voxel u,v;

  hscn = CopyScene(scn);
  A    = Spheric(1.0);
  for (u.z=0; u.z < hscn->zsize; u.z++){
    for (u.y=0; u.y < hscn->ysize; u.y++){
      for (u.x=0; u.x < hscn->xsize; u.x++){
	p = u.x + hscn->tby[u.y] + hscn->tbz[u.z];
	for (i=1; i < A->n; i++){
	  v.x = u.x + A->dx[i];
	  v.y = u.y + A->dy[i];
	  v.z = u.z + A->dz[i];
	  if (ValidVoxel(hscn,v.x,v.y,v.z)){
	    q = v.x + hscn->tby[v.y] + hscn->tbz[v.z];
	    if (label->data[p] < label->data[q]){
	      hscn->data[p] = value;
	      break;
	    }
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  
  return(hscn);
}

int ScenesAreEqual(Scene *scn1, Scene *scn2)
{
  int i,n;

  if((scn1->xsize != scn2->xsize)||(scn1->ysize != scn2->ysize)||(scn1->zsize != scn2->zsize)) {
    fprintf(stderr,"Scenes have different sizes\n");
    return(0);
  }
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for(i=0;i<n;i++) {
    if(scn1->data[i] != scn2->data[i]) {
      fprintf(stderr,"Scenes have different values (in voxel %d for example)\n",i);
      return(0);
    }
  }
  return(1);
}

void CompareScenes(Scene *scn1, Scene *scn2)
{
  int i,n,nvoxel=0;
  //  int diff;
  //  long sum=0;
  //  float mdiff;

  if((scn1->xsize != scn2->xsize)||(scn1->ysize != scn2->ysize)||(scn1->zsize != scn2->zsize)) {
    fprintf(stderr,"Scenes have different sizes\n");
    return;
  }
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for(i=0;i<n;i++) {
    if(scn1->data[i] != scn2->data[i]) {
      //  fprintf(stderr,"Scenes have different values (%d and %d) in voxel %d\n",scn1->data[i],scn2->data[i],i);
      //      diff = scn1->data[i] - scn2->data[i];
      //      if(diff<0) diff = -diff;
      //      sum += diff;
      nvoxel++;
    }
  }
  //  mdiff = (float) sum/nvoxel;
  printf("%d voxels are different in a scene of %d voxels\n",nvoxel,n);
  //  printf("the mean difference for different voxels is %f\n", mdiff);
  return;
}

Scene *GetBorder3(Scene *scn, AdjRel3 *A)
{
  Scene *hscn=NULL;
  int p,q,i;
  Voxel u,v;

  hscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  for (u.z=0; u.z < hscn->zsize; u.z++){
    for (u.y=0; u.y < hscn->ysize; u.y++){
      for (u.x=0; u.x < hscn->xsize; u.x++){
	p = u.x + hscn->tby[u.y] + hscn->tbz[u.z];
	if (scn->data[p] != 0) {
	  for (i=1; i < A->n; i++){
	    v.x = u.x + A->dx[i];
	    v.y = u.y + A->dy[i];
	    v.z = u.z + A->dz[i];
	    if (ValidVoxel(hscn,v.x,v.y,v.z)){
	      q = v.x + hscn->tby[v.y] + hscn->tbz[v.z];
	      if (scn->data[p] != scn->data[q]){
		hscn->data[p] = scn->data[p];
	        break;
	      }
	    } else {
	      hscn->data[p] = scn->data[p];
	      break;
	    }
	  }
	}
      }
    }
  }
  return(hscn);
}

void DrawAdj3(Scene *scn, AdjRel3 *A, Voxel *center, int value) {

  int i,j=0;
  Voxel v,c;

  if (center == NULL) {
    c.x = scn->xsize/2;
    c.y = scn->ysize/2;
    c.z = scn->zsize/2;
  } else {
    c.x = center->x;
    c.y = center->y;
    c.z = center->z;
  }
  for (i=0;i<A->n;i++) {
    v.x = c.x + A->dx[i];
    v.y = c.y + A->dy[i];
    v.z = c.z + A->dz[i];
    if (ValidVoxel(scn,v.x,v.y,v.z)) {
      j= scn->tbz[v.z]+scn->tby[v.y]+v.x;
      scn->data[j] = value;
    }    
  }
}


Scene *ShapeBasedInterp(Scene *scn,float dx,float dy,float dz){

  Scene *iscn=NULL,*dscn,*obj,*idist,*bin;
  Image *dist,*img;
  Voxel v;
  AdjRel *A = Circular(1.5);
  int n,i,j,k;
  float min = (float)INT_MAX;
  Curve *c;


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
    return (CopyScene(scn));
  } 

  //  fprintf(stderr,"Starting interpolation\n");

  //iscn = CreateScene ((int)((float)(scn->xsize-1)*scn->dx/dx)+1, (int)((float)(scn->ysize-1)*scn->dy/dy)+1, (int)((float)(scn->zsize-1)*scn->dz/dz)+1);
  iscn = CreateScene(ROUND((float)(scn->xsize)*scn->dx/dx),
		     ROUND((float)(scn->ysize)*scn->dy/dy),
		     ROUND((float)(scn->zsize)*scn->dz/dz));


  //fprintf(stderr,"New size: %d %d %d\n",iscn->xsize,iscn->ysize,iscn->zsize);


  c = SceneLabels(scn);
  //  fprintf(stderr,"Total of %d objects\n",c->n-1);

  for (k=0;k<c->n;k++) {
    i = (int)c->X[k];
    /* Avoid background */
    if (i) {
      //fprintf(stderr,"Interpolating object %d\n",i);
    bin = Threshold3(scn,i,i);

    if (dx != scn->dx) {
      //fprintf(stderr,"Interpolation in x\n");
      dscn = CreateScene(bin->xsize,bin->ysize,bin->zsize);
      dscn->dx=scn->dx;
      dscn->dy=scn->dy;
      dscn->dz=scn->dz;
      for (v.x = 0; v.x <bin->xsize;v.x++) {
	img = GetXSlice(bin,v.x);
	dist = SignedDistTrans(img,A,BOTH);
	PutXSlice(dist,dscn,v.x);
	DestroyImage(&img);
	DestroyImage(&dist);
      }
      DestroyScene(&bin);
      idist = LinearInterp(dscn,dx,scn->dy,scn->dz);
      DestroyScene(&dscn);

      obj = Threshold3(idist,0,INT_MAX);
      DestroyScene(&idist);
      bin = obj;
    }

    if (dy != scn->dy) {
      //fprintf(stderr,"Interpolation in y\n");
      dscn = CreateScene(bin->xsize,bin->ysize,bin->zsize);
      dscn->dx=scn->dx;
      dscn->dy=scn->dy;
      dscn->dz=scn->dz;
      for (v.y = 0; v.y <bin->ysize;v.y++) {
	img = GetYSlice(bin,v.y);
	dist = SignedDistTrans(img,A,BOTH);
	PutYSlice(dist,dscn,v.y);
	DestroyImage(&img);
	DestroyImage(&dist);
      }
      DestroyScene(&bin);
      idist = LinearInterp(dscn,scn->dx,dy,scn->dz);
      DestroyScene(&dscn);
      obj = Threshold3(idist,0,INT_MAX);
      DestroyScene(&idist);
      bin = obj;
    }

    if (dz != scn->dz) {
      //fprintf(stderr,"Interpolation in z\n");
      dscn = CreateScene(bin->xsize,bin->ysize,bin->zsize);
      dscn->dx=scn->dx;
      dscn->dy=scn->dy;
      dscn->dz=scn->dz;
      for (v.z = 0; v.z <bin->zsize;v.z++) {
	img = GetSlice(bin,v.z);
	dist = SignedDistTrans(img,A,BOTH);
	PutSlice(dist,dscn,v.z);
	DestroyImage(&img);
	DestroyImage(&dist);
      }
      DestroyScene(&bin);
      idist = LinearInterp(dscn,scn->dx,scn->dy,dz);
      DestroyScene(&dscn);
      obj = Threshold3(idist,0,INT_MAX);
      DestroyScene(&idist);
      bin = obj;
    }
    //fprintf(stderr,"Size: %d x %d x %d.\n",bin->xsize,bin->ysize,bin->zsize);
    //fprintf(stderr,"Restoring labels...\n");

    if(iscn->xsize != bin->xsize || 
       iscn->ysize != bin->ysize || 
       iscn->zsize != bin->zsize)
      Warning("Inconsistent sizes","ShapeBasedInterp");

    n = iscn->xsize * iscn->ysize * iscn->zsize;

    for(j=0;j<n;j++) 
      if(bin->data[j])
	iscn->data[j] = i;
    DestroyScene(&bin);
    }
  }
  iscn->dx = dx;
  iscn->dy = dy;
  iscn->dz = dz;
  DestroyCurve(&c);
  DestroyAdjRel(&A);
  return(iscn);
}

Scene *MergeLabels(Scene *scn1, Scene *scn2) {

  Scene *scn,*scn3;
  int maxlabel=0,i,n;

  maxlabel = MaximumValue3(scn1);
  n = scn2->xsize * scn2->ysize * scn2->zsize;
  scn = CreateScene(scn2->xsize,scn2->ysize,scn2->zsize);
  for(i=0;i<n;i++) 
   if(scn2->data[i])
     scn->data[i] = scn2->data[i] + maxlabel;
  scn3 = Or3(scn1,scn);
  DestroyScene(&scn);
  return(scn3);
}

Scene *Image2Scene(Image *img) {
  Scene *scn;
  int i,n,u1,z;
  Pixel u;
  scn = CreateScene(img->ncols,img->nrows,256);
  n = img->nrows*img->ncols*256;
  for (i=0;i<n;i++)
    scn->data[i] = 0;
  for(u.y=0;u.y < img->nrows;u.y++)
    for(u.x=0;u.x < img->ncols;u.x++) {
      u1  = u.x + img->tbrow[u.y];
      for(z=0;z<img->val[u1];z++)
	scn->data[VoxelAddress(scn,u.x,u.y,z)] = img->val[u1];
    }
  scn->maxval = MaximumValue3(scn);
  return scn;
}

Scene *Image2BinaryScene(Image *img,int border) { // Constroi uma Scene binária com borda a partir de uma imagem em níveis de cinza, usando o brilho como dimensão Z
  Scene *scn;
  int i,n,u1,z;
  Pixel u;
  int max = MaximumValue(img);
  scn = CreateScene(img->ncols+(2*border), img->nrows+(2*border), max+(3*border));
  n = (img->ncols+(2*border))*(img->nrows+(2*border))*(max+(3*border)); // garante espaço tamanho borda para crescer para os lados, para cima e para baixo e para dentro (em z tenho 3*border esticando o espaço interno do volume)
  for (i=0;i<n;i++)
    scn->data[i] = 0;
  for(u.y=0;u.y < img->nrows;u.y++)
    for(u.x=0;u.x < img->ncols;u.x++) {
      u1  = u.x + img->tbrow[u.y];
      for(z=0;z<img->val[u1]+border;z++)
	scn->data[VoxelAddress(scn,u.x+border,u.y+border,max-1-z+border)] = 1;
    }
  scn->maxval = MaximumValue3(scn);
  return scn;
}


Scene*  SubScene(Scene *scn, int x0, int x1, int y0, int y1, int z0, int z1)
{
  Scene *sub = NULL;
  int x, y, z, p, q;
  if( ( ( x0 > x1 ) && ( x1 != -1 ) ) || ( ( y0 > y1 ) && ( y1 != -1 ) ) || ( ( z0 > z1 ) && ( z1 != -1 ) ) ) {
    Error(MSG1,"SubScene: Use x0 <= x1, y0 <= y1, and z0 <= z1." );
  }
  if( ( scn->xsize <= x1 ) || ( scn->ysize <= y1 ) || ( scn->zsize <= z1 ) ) {
    Error(MSG1,"SubScene: Box dimensions are greater than input images.");
  }
  sub = CreateScene( x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1 );
  if( x1 == -1 ) x1 = scn->xsize - 1;
  if( y1 == -1 ) y1 = scn->ysize - 1;
  if( z1 == -1 ) z1 = scn->zsize - 1;
  for( z = z0; z <= z1; z++ ) {
    for( y = y0; y <= y1; y++ ) {
      for( x = x0; x <= x1; x++ ) {
	p = VoxelAddress( scn, x, y, z );
	q = VoxelAddress( sub, x - x0, y - y0, z - z0 );
	sub->data[ q ] = scn->data[ p ];
      }
    }
  }
  return( sub );
}

Scene*  Rotate3(Scene *scn, 
		double thx, double thy, double thz, // angles to rotate
		float cx, float cy, float cz) // center of the rotation
{
  int p;
  Voxel u;
  Scene *res;
  RealMatrix *trans1,*rot1,*rot2,*rot3,*trans2,*aux1,*aux2,*inv;
  float vx,vy,vz;
  //RealMatrix *vox;
  trans1 = TranslationMatrix3(-cx,-cy,-cz);
  rot1 = RotationMatrix3(0,thx);
  rot2 = RotationMatrix3(1,thy);
  rot3 = RotationMatrix3(2,thz);
  trans2 = TranslationMatrix3(cx,cy,cz);

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
	vx = inv->val[0][0]*u.x + inv->val[0][1]*u.y + inv->val[0][2]*u.z + inv->val[0][3];
	vy = inv->val[1][0]*u.x + inv->val[1][1]*u.y + inv->val[1][2]*u.z + inv->val[1][3];
	vz = inv->val[2][0]*u.x + inv->val[2][1]*u.y + inv->val[2][2]*u.z + inv->val[2][3];
	if ((vx<=res->xsize-1) && (vy<=res->ysize-1) && (vz<=res->zsize-1) && vx>0 && vy>0 && vz>0)
	  {
	    res->data[p]=GetVoxelValue_trilinear(scn,vx,vy,vz);
	  }
	else
	  res->data[p]=0;
	/* old version
	vox = TransformVoxel(inv,u);
	if ((vox->val[0][0]<=res->xsize-1)&&(vox->val[0][1]<=res->ysize-1)
	    &&(vox->val[0][2]<=res->zsize-1 && vox->val[0][0]>0 && vox->val[0][1]>0 && vox->val[0][2]>0))
	  {
	    res->data[p]=GetVoxelValue_trilinear(scn,vox->val[0][0],vox->val[1][0],vox->val[2][0]);
	  }
	else
	  res->data[p]=0;
	DestroyRealMatrix(&vox);
	*/
      }
  DestroyRealMatrix(&inv);
  res->dx=scn->dx;
  res->dy=scn->dy;
  res->dz=scn->dz;
  return res;
}



// Computes the value of the point (x,y,z) using trilinear interpolation
float GetVoxelValue_trilinear(Scene *scn, float x, float y, float z)
{
  int xsize,ysize,zsize;
  int px,py,pz,i1,i2;
  float dx,dy,dz;
  int nbx=0,nby=0,nbz=0;
  double p1,p2,p3,p4,p5,p6,res;
  xsize=scn->xsize;
  ysize=scn->ysize;
  zsize=scn->zsize;
  if (x>xsize-1||y>ysize-1||z>zsize-1||x<0||y<0||z<0) {
    //printf("GetVoxelValue_trilinear out-of-bounds\n"); 
    return(0);
  }
  px = (int)x;  dx=x-px;
  py = (int)y;  dy=y-py;
  pz = (int)z;  dz=z-pz;

  if (px<xsize-1) nbx=1; // If it's not on the border, it has a neighour ahead
  if (py<ysize-1) nby=1;
  if (pz<zsize-1) nbz=1;

  // 1st: Interpolate in Z
  i1 = GetVoxel(scn,VoxelAddress(scn,px,py,pz));
  i2 = GetVoxel(scn,VoxelAddress(scn,px,py,pz+nbz));
  p1 = i1 + dz*(i2-i1);
  i1 = GetVoxel(scn,VoxelAddress(scn,px+nbx,py,pz));
  i2 = GetVoxel(scn,VoxelAddress(scn,px+nbx,py,pz+nbz));
  p2 = i1 + dz*(i2-i1);
  i1 = GetVoxel(scn,VoxelAddress(scn,px,py+nby,pz));
  i2 = GetVoxel(scn,VoxelAddress(scn,px,py+nby,pz+nbz));
  p3 = i1 + dz*(i2-i1);
  i1 = GetVoxel(scn,VoxelAddress(scn,px+nbx,py+nby,pz));
  i2 = GetVoxel(scn,VoxelAddress(scn,px+nbx,py+nby,pz+nbz));
  p4 = i1 + dz*(i2-i1);
  // 2nd: Interpolate in X
  p5 = p1 + dx*(p2-p1);
  p6 = p3 + dx*(p4-p3);
  // 3rd: Interpolate in Y
  res = p5 + dy*(p6-p5);
  return (int)(res+0.5);
}



// return the nearest voxel
int GetVoxelValue_nn(Scene *scn, float x, float y, float z)
{
  Voxel v;
  //  float zero=0;
  if (x>scn->xsize-1||y>scn->ysize-1||z>scn->zsize-1||x<(float)0||y<(float)0||z<(float)0) {
    printf("GetVoxelValue_nn out-of-bounds - p(%f,%f,%f) - max (%d,%d,%d)\n",x,y,z, scn->xsize-1, scn->ysize-1, scn->zsize-1); exit(1);
  }
  v.x=(int)(x+0.5);
  v.y=(int)(y+0.5);
  v.z=(int)(z+0.5);
  return GetVoxel(scn,VoxelAddress(scn,v.x,v.y,v.z));;
}

Scene *ReadScene_Nifti1( char filename[ ] ) {
  nifti_image *nii;
  Scene *scn;
  float max;
  int p;
  
  nii = nifti_image_read( filename , 1 );
  scn = CreateScene( nii->nx, nii->ny, nii->nz );
  scn->dx = nii->dx;
  scn->dy = nii->dy;
  scn->dz = nii->dz;
  scn->nii_hdr = nii;

  if( ( nii->datatype == NIFTI_TYPE_COMPLEX64 ) ||
      ( nii->datatype == NIFTI_TYPE_FLOAT64 ) ||
      ( nii->datatype == NIFTI_TYPE_RGB24 ) ||
      ( nii->datatype == NIFTI_TYPE_RGB24 ) ||
      ( nii->datatype >= NIFTI_TYPE_UINT32 ) ||
      ( nii->dim[ 0 ] < 3 ) || ( nii->dim[ 0 ] > 4 ) ) {
    printf( "Error: Data format not supported, or header is corrupted.\n" );
    printf( "Data that is NOT supported: complex, double, RGB, unsigned integer of 32 bits, temporal series and statistical images.\n" );
    exit( -1 );
  }
  
  if( nii->datatype == NIFTI_TYPE_INT32 ) {
    //printf( "Integer src image.\n" );
    memcpy( scn->data, nii->data, nii->nvox * nii->nbyper );
  }
  if( ( nii->datatype == NIFTI_TYPE_INT16 ) || ( nii->datatype == NIFTI_TYPE_UINT16 ) ) {
    //printf( "Short src image.\n" );
    for( p = 0; p < scn->n; p++ ) {
      scn->data[ p ] = ( ( unsigned short* ) nii->data )[ p ];
    }
  }
  if( ( nii->datatype == NIFTI_TYPE_INT8 ) || ( nii->datatype == NIFTI_TYPE_UINT8 ) ) {
    //printf( "Char src image.\n" );
    for( p = 0; p < scn->n; p++ ) {
      scn->data[ p ] = ( ( unsigned char* ) nii->data )[ p ];
    }
  }
  if( nii->datatype == NIFTI_TYPE_FLOAT32 ) {
    //printf( "Float src image.\n" );
    // max used to set data to integer range without loosing its precision.
    max = 0.0;
    for( p = 0; p < scn->n; p++ ) {
      if( max < ( ( float* ) nii->data )[ p ] ) {
	max = ( ( float* ) nii->data )[ p ];
      }
    }
    nii->flt_conv_factor = 10000.0 / max;
    
    for( p = 0; p < scn->n; p++ ) {
      scn->data[ p ] = ROUND( ( ( float* ) nii->data )[ p ] * nii->flt_conv_factor );
    }
  }
  // apply the "descompression"
  for (int p = 0; p < scn->n; p++) {
    scn->data[p] = ROUND((scn->data[p] * nii->scl_slope) + nii->scl_inter);
  }
  
  free( nii->data );
  nii->data = NULL;
  return( scn );
}

void WriteScene_Nifti1( Scene *scn, char filename[ ] ) {
  nifti_image *nii;
  int len;
  int p;

  nii = scn->nii_hdr;
  if( nii == NULL )
    Error( MSG4, "WriteScene_Nifti1" );

  if( nii->data != NULL )
    free( nii->data );
  nii->data = calloc( nii->nbyper, nii->nvox );

  if( nii->datatype == NIFTI_TYPE_INT32 ) {
    //printf( "Integer src image.\n" );
    memcpy( nii->data, scn->data, nii->nvox * nii->nbyper );
  }
  else if( ( nii->datatype == NIFTI_TYPE_INT16 ) || ( nii->datatype == NIFTI_TYPE_UINT16 ) ) {
    //printf( "Short src image.\n" );
    for( p = 0; p < scn->n; p++ ) {
      ( ( unsigned short* ) nii->data )[ p ] = scn->data[ p ];
    }
  }
  else if( ( nii->datatype == NIFTI_TYPE_INT8 ) || ( nii->datatype == NIFTI_TYPE_INT8 ) || ( nii->datatype == NIFTI_TYPE_UINT8 ) ) {
    //printf( "Char src image.\n" );
    for( p = 0; p < scn->n; p++ ) {
      ( ( unsigned char* ) nii->data )[ p ] = scn->data[ p ];
    }
  }
  else if( nii->datatype == NIFTI_TYPE_FLOAT32 ) {
    //printf( "Float src image.\n" );
    // max used to set data to integer range without loosing its precision.
    for( p = 0; p < scn->n; p++ ) {
      ( ( float* ) nii->data )[ p ] = scn->data[ p ] / nii->flt_conv_factor;
    }
  }
  else 
    Error( MSG5, "WriteScene_Nifti1" );

  len = strlen( filename );
  if( ( strcasecmp( filename + len - 4, ".hdr" ) == 0 ) || ( strcasecmp( filename + len - 4, ".img" ) == 0 ) )
    nii->nifti_type = NIFTI_FTYPE_NIFTI1_2;
  else
    nii->nifti_type = NIFTI_FTYPE_NIFTI1_1;
  nifti_set_filenames( nii, filename, 0, nii->byteorder );
  
  nifti_image_write( nii );
  free( nii->data );
  nii->data = NULL;
}

void CloneNiftiHeader( Scene *src, Scene *dst ) {
  if( ( src != NULL ) && ( dst != NULL ) && ( src->nii_hdr != NULL ) ) {
    if( dst->nii_hdr != NULL ) {
      nifti_image_free( dst->nii_hdr );
      dst->nii_hdr = NULL;
    }
    dst->nii_hdr = nifti_copy_nim_info( src->nii_hdr );
  }
}

void CopySceneHeader( Scene *src, Scene *dst ) {
  dst->dx  = src->dx;
  dst->dy  = src->dy;
  dst->dz  = src->dz;
  CloneNiftiHeader( src, dst );
}
