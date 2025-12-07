#include "adjacency.h"
#include "adjacency3.h"
#include "analysis3.h"
#include "classify.h"
#include "common.h"
#include "geometry.h"
#include "mathematics3.h"
#include "normal.h"
#include "radiometric.h"
#include "segmentation3.h"
#include "shading.h"
#include "shell.h"
#include "analysis3.h"
#include "mathematics.h"

Shell *CreateShell (Scene *scn, Scene *alpha, int w) {

  Scene *tmp,*border;
  int i,n,nvoxels;
  Shell *shell;
  AdjRel3 *A;

  n = scn->xsize*scn->ysize*scn->zsize;
  A = Spheric(1.8);
  border = Label2Border(scn,A,INTERIOR,w*w);
  tmp = NonTransparentVoxels(border,alpha);
  DestroyScene(&border);
  nvoxels=0; 

  for (i=0;i<n;i++)  
    if (tmp->data[i] != 0) 
      nvoxels++;  

  shell = NewShell(tmp->xsize,tmp->ysize,tmp->zsize,nvoxels);
  
  SetShellVoxels(shell,tmp);
  SetShellList(shell,tmp);
  shell->maxval = MaximumValue3(scn);
  shell->nobjs = MaximumValue3(tmp);
  DestroyScene(&tmp);
  SetShellLabel(shell,scn);
  SetShellOpacity(shell,alpha);
  SetDistNormal (shell);
  shell->dx  = scn->dx;
  shell->dy  = scn->dy;
  shell->dz  = scn->dz;
  DestroyAdjRel3(&A);
  return(shell);
}

Shell *CreateBodyShell (Scene *scn, Scene *alpha) {

  Scene *tmp;
  int i,n,nvoxels;
  Shell *shell;

  n = scn->xsize * scn->ysize * scn->zsize;
  tmp = NonTransparentVoxels(scn,alpha);

  nvoxels=0; 
  for (i=0;i<n;i++)  
    if (tmp->data[i] != 0) 
      nvoxels++;  

  shell = NewBodyShell(tmp->xsize,tmp->ysize,tmp->zsize,nvoxels);
  
  SetShellVoxels(shell,tmp);
  SetShellList(shell,tmp);
  SetShellOpacity(shell,alpha);
  SetBodyShellNormal(shell,tmp,Gradient3);
  SetBodyShellValue (shell,tmp);

  shell->dx  = scn->dx;
  shell->dy  = scn->dy;
  shell->dz  = scn->dz;
  shell->maxval = MaximumValue3(scn);
  DestroyScene(&tmp);
  return(shell);
}

Scene *NonTransparentVoxels (Scene *scn, Scene *alpha) 
{
  Scene *out;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;
  int i,j,n,aux=0,xysize=0;
  Voxel v,q;

  xysize = scn->xsize*scn->ysize;
  A=Spheric(1.0);
  V=AdjVoxels(scn,A);
  out = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  n = scn->xsize*scn->ysize*scn->zsize;
  for (i=0;i<n;i++) {
    if (scn->data[i] != 0) {
      if (alpha->data[i] > 12) {
	aux = i%xysize;
	v.x = aux%scn->xsize;
	v.y = aux/scn->xsize;
	v.z = i/xysize;
	for (j=0;j<A->n;j++) {
          q.x = v.x + A->dx[j];
          q.y = v.y + A->dy[j];
          q.z = v.z + A->dz[j];
 	  if (ValidVoxel(scn, q.x, q.y, q.z)) {
	    if (alpha->data[V->dp[j]] < 242) {
              out->data[i] = scn->data[i];
              break;
	    } 
	  } else {
              out->data[i] = scn->data[i];
	  }
	}
      }
    }
  }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
  return(out);
}

void    DestroyShell(Shell **shell) {

  Shell *aux = *shell;
  if(aux != NULL){
    if (aux->voxel    != NULL)  free(aux->voxel); 
    if (aux->pointer  != NULL)  free(aux->pointer);
    if (aux->normaltable  != NULL)  free(aux->normaltable);
    if (aux->body  != NULL)  free(aux->body);
    DestroyImage(&(aux->Myz));
    DestroyImage(&(aux->Mzx));
    free(aux);    
    *shell = NULL;
  }

}



Shell *ReadShell (char *filename) {

  FILE   *fp=NULL;
  Shell *shell;
  char type[10];

  shell=(Shell *)calloc(1, sizeof(Shell));

  fp = fopen(filename,"rb"); 
  if (fp == NULL) 
    Error(MSG2,"ReadShell");

  fscanf(fp,"%s\n",type);
  fscanf(fp,"%d %d %d\n",&shell->xsize,&shell->ysize,&shell->zsize);
  fscanf(fp,"%f %f %f\n",&shell->dx,&shell->dy,&shell->dz);
  fscanf(fp,"%d\n",&shell->nvoxels);
  shell->voxel = ReadVoxels (fp,shell->nvoxels);
  shell->pointer = ReadList (fp,shell->nvoxels);
  shell->Myz = ReadM(fp,shell->ysize,shell->zsize);
  shell->Mzx = ReadM(fp,shell->zsize,shell->xsize);
  shell->normaltable = CreateNormalTable();
  shell->body = NULL;
  GetShellMaximum(shell);
  fclose(fp);
  return(shell);
}

Shell *ReadBodyShell (char *filename) {

  FILE   *fp=NULL;
  Shell *shell;
  char type[10];

  shell=(Shell *)calloc(1, sizeof(Shell));

  fp = fopen(filename,"rb"); 
  if (fp == NULL) 
    Error(MSG2,"ReadBodyShell");

  fscanf(fp,"%s\n",type);
  fscanf(fp,"%d %d %d\n",&shell->xsize,&shell->ysize,&shell->zsize);
  fscanf(fp,"%f %f %f\n",&shell->dx,&shell->dy,&shell->dz);
  fscanf(fp,"%d\n",&shell->nvoxels);
  shell->voxel = ReadVoxels (fp,shell->nvoxels);
  shell->pointer = ReadList (fp,shell->nvoxels);
  shell->Myz = ReadM(fp,shell->ysize,shell->zsize);
  shell->Mzx = ReadM(fp,shell->zsize,shell->xsize);
  shell->body = ReadBody (fp,shell->nvoxels);
  shell->normaltable = CreateNormalTable();
  GetShellMaximum(shell);
  fclose(fp);
  return(shell);
}

Attrib *ReadVoxels (FILE *fp, int n) {

  int i;
  Attrib *voxel;
  uchar *data8= NULL;
  ushort *data16 =NULL;

  voxel=(Attrib *)calloc(n, sizeof(Attrib));

  data8 = AllocUCharArray(n);

  fread(data8,sizeof(uchar),n,fp);
  for (i=0; i < n; i++)
    voxel[i].obj = data8[i];
  fread(data8,sizeof(uchar),n,fp);
  for (i=0; i < n; i++)
    voxel[i].opac = data8[i];
  fread(data8,sizeof(uchar),n,fp);
  for (i=0; i < n; i++)
    voxel[i].visible = data8[i];

  free(data8);

  data16 = AllocUShortArray(n);

  fread(data16,sizeof(ushort),n,fp);
  for (i=0; i < n; i++)
    voxel[i].x = data16[i];

  fread(data16,sizeof(ushort),n,fp);
  for (i=0; i < n; i++)
    voxel[i].normal = data16[i];

  free(data16);

  return(voxel);
}

PAttrib *ReadList (FILE *fp, int n) {

  int i;
  PAttrib *list;
  ushort *data16=NULL;
  int *data32=NULL;

  list=(PAttrib *)calloc(n, sizeof(PAttrib));

  data16 = AllocUShortArray(n);

  fread(data16,sizeof(ushort),n,fp);
  for (i=0; i < n; i++)
    list[i].y = data16[i];

  free(data16);

  data32 = AllocIntArray(n);

  fread(data32,sizeof(int),n,fp);
  for (i=0; i < n; i++)
    list[i].i = data32[i];

  free(data32);

  return(list);
}

CAttrib *ReadBody (FILE *fp, int n) {

  int i;
  CAttrib *body;
  ushort *data16=NULL;
  int *data32=NULL;

  body=(CAttrib *)calloc(n, sizeof(CAttrib));

  data16 = AllocUShortArray(n);

  fread(data16,sizeof(ushort),n,fp);
  for (i=0; i < n; i++)
    body[i].normalmag = data16[i];

  free(data16);

  data32 = AllocIntArray(n);

  fread(data32,sizeof(int),n,fp);
  for (i=0; i < n; i++)
    body[i].val = data32[i];

  free(data32);

  return(body);
}

Image *ReadM(FILE *fp, int ncols, int nrows)
{
  int n;
  Image *img;

  img = CreateImage(ncols,nrows+1);
  n = ncols*(nrows+1);

  fread(img->val,sizeof(int),n,fp);

  return(img);
}

void WriteShell (Shell *shell, char *filename) {

  FILE   *fp=NULL;

  fp = fopen(filename,"wb"); 
  if (fp == NULL) 
    Error(MSG2,"WriteScene");

  fprintf(fp,"SH\n");
  fprintf(fp,"%d %d %d\n",shell->xsize,shell->ysize,shell->zsize);
  fprintf(fp,"%f %f %f\n",shell->dx,shell->dy,shell->dz);
  fprintf(fp,"%d\n",shell->nvoxels);
  WriteVoxels (fp,shell->voxel,shell->nvoxels);
  WriteList (fp,shell->pointer,shell->nvoxels);
  WriteM (fp,shell->Myz);
  WriteM (fp,shell->Mzx);
  fclose(fp);
}

void WriteBodyShell (Shell *shell, char *filename) {

  FILE   *fp=NULL;

  fp = fopen(filename,"wb"); 
  if (fp == NULL) 
    Error(MSG2,"WriteScene");

  fprintf(fp,"SH\n");
  fprintf(fp,"%d %d %d\n",shell->xsize,shell->ysize,shell->zsize);
  fprintf(fp,"%f %f %f\n",shell->dx,shell->dy,shell->dz);
  fprintf(fp,"%d\n",shell->nvoxels);
  WriteVoxels (fp,shell->voxel,shell->nvoxels);
  WriteList (fp,shell->pointer,shell->nvoxels);
  WriteM (fp,shell->Myz);
  WriteM (fp,shell->Mzx);
  WriteBody (fp,shell->body,shell->nvoxels);
  fclose(fp);
}

void WriteVoxels (FILE *fp, Attrib *voxel, int n) {

  int i;
  uchar  *data8 =NULL;
  ushort *data16=NULL;

  data8 = AllocUCharArray(n);

  for (i=0; i < n; i++)
    data8[i] = voxel[i].obj;
  fwrite(data8,sizeof(uchar),n,fp);

  for (i=0; i < n; i++)
    data8[i] = voxel[i].opac;
  fwrite(data8,sizeof(uchar),n,fp);

  for (i=0; i < n; i++)
    data8[i] = voxel[i].visible;
  fwrite(data8,sizeof(uchar),n,fp);

  free(data8);

  data16 = AllocUShortArray(n);

  for (i=0; i < n; i++)
    data16[i] = voxel[i].x;
  fwrite(data16,sizeof(ushort),n,fp);

  for (i=0; i < n; i++)
    data16[i] = voxel[i].normal;
  fwrite(data16,sizeof(ushort),n,fp);

  free(data16);

}

void WriteList (FILE *fp, PAttrib *list, int n) {

  int i;
  ushort *data16=NULL;
  int *data32 =NULL;

  data16 = AllocUShortArray(n);

  for (i=0; i < n; i++)
    data16[i] = list[i].y;
  fwrite(data16,sizeof(ushort),n,fp);

  free(data16);
  
  data32 = AllocIntArray(n);

  for (i=0; i < n; i++)
    data32[i] = list[i].i;
  fwrite(data32,sizeof(int),n,fp);

  free(data32);
}

void WriteBody (FILE *fp, CAttrib *body, int n) {

  int i;
  ushort *data16=NULL;
  int *data32 =NULL;

  data16 = AllocUShortArray(n);

  for (i=0; i < n; i++)
    data16[i] = body[i].normalmag;
  fwrite(data16,sizeof(ushort),n,fp);

  free(data16);
  
  data32 = AllocIntArray(n);

  for (i=0; i < n; i++)
    data32[i] = body[i].val;
  fwrite(data32,sizeof(int),n,fp);

  free(data32);
}

void WriteM(FILE *fp, Image *img)
{
  int n;

  n = img->ncols*img->nrows;
  fwrite(img->val,sizeof(int),n,fp);
}

Shell *NewShell (int xsize, int ysize, int zsize, int nvoxels) {

  Shell *shell=NULL;

  shell=(Shell *)calloc(1, sizeof(Shell));
  if (shell == NULL){
    Error(MSG1,"CreateShell");
  }

  shell->xsize = xsize;
  shell->ysize = ysize;
  shell->zsize = zsize;
  shell->nvoxels = nvoxels;
  shell->scn = NULL;


  shell->voxel=(Attrib *)calloc(shell->nvoxels, sizeof(Attrib));
  if (shell->voxel == NULL) {
    Error(MSG1,"NewShell");
  }

  shell->body = NULL;

  shell->pointer=(PAttrib *)calloc(shell->nvoxels, sizeof(PAttrib));
  if (shell->pointer == NULL) {
    Error(MSG1,"NewShell");
  }

  /* Myz and Mzx */

  shell->Myz=CreateImage(ysize, zsize+1);
  shell->Mzx=CreateImage(zsize, xsize+1);  

  SetImage(shell->Myz,-1);
  SetImage(shell->Mzx,-1);

  shell->Mzx->val[zsize*xsize] = shell->nvoxels;
  shell->Myz->val[zsize*ysize] = shell->nvoxels;

  shell->normaltable = CreateNormalTable();
  shell->nvoxels = nvoxels;
  shell->dx  = 0.0;
  shell->dy  = 0.0;
  shell->dz  = 0.0;  

  return(shell);
}

Shell *NewBodyShell (int xsize, int ysize, int zsize, int nvoxels) {

  Shell *shell=NULL;

  shell= NewShell (xsize,ysize,zsize,nvoxels);
  
  shell->body=(CAttrib *)calloc(shell->nvoxels, sizeof(CAttrib));
  if (shell->body == NULL) {
    Error(MSG1,"NewBodyShell");
  }
  return (shell);
}

Scene *InnerBorder(Scene *bin, int w) {

  int w2 = w*w;
  Scene *dist,*border;
  AdjRel3 *A = Spheric(1.8);

  dist = TDistTrans3(bin,A,BOTH,w2,0);
  border = Threshold3(dist,0,w2);  
  DestroyScene(&dist);
  DestroyAdjRel3(&A);
  return (border);
}


Shell *Object2Shell (Scene *scn, int w) {

  Shell *shell=NULL;
  int i,n,nvoxels,w2;
  Scene *bin=NULL, *border=NULL, *dist=NULL;
  Curve *c,*l;
  AdjRel3 *A = Spheric(1.8);
  
  n = scn->xsize * scn->ysize * scn->zsize; 
  w2 = w*w;

  c = CreateCurve(2);
  c->X[0] = 0.0; 
  c->Y[0] = 0.5;
  c->X[1] = (double)ROUND((float)w2);
  c->Y[1] = 1.0;

  border = Label2Border(scn,A,BOTH,w2);

  nvoxels=0;
  for (i=0;i<n;i++)  
    if (border->data[i] != 0) 
      nvoxels++;

  shell = NewShell(scn->xsize,scn->ysize,scn->zsize,nvoxels);
  SetShellVoxels (shell,border);
  SetShellList (shell,border);

  l = ShellObjHistogram(shell);

  for(i=0;i<l->n;i++) {
    if (l->Y[i] > 0) {
      bin = Threshold3(border,i,i);
      dist = TDistTrans3(bin,A,BOTH,w2,0);
      DestroyScene(&bin);
      SetObjectNormal(shell,dist,DistGradient3,i);
      ClassifyObject(shell,dist,c,DistGradient3,i);
      DestroyScene(&dist);
    }
  }
  DestroyScene(&border);

  shell->dx  = scn->dx;
  shell->dy  = scn->dy;
  shell->dz  = scn->dz;
  shell->maxval = MaximumValue3(scn);
  shell->nobjs  = GetShellNObjects(shell);
  DestroyCurve(&c);
  DestroyCurve(&l);
  DestroyAdjRel3(&A);
  return shell;
}

Shell *MergeShell(Shell *shell1, Shell *shell2) {

  Scene *scn,*scn2,*scn1;
  Shell *shell;

  scn1 = LabelScene(shell1);
  scn2 = LabelScene(shell2);
  scn = MergeLabels(scn1,scn2);
  DestroyScene(&scn1); 
  DestroyScene(&scn2); 
  shell = Object2Shell(scn,5);
  DestroyScene(&scn); 
    
  return(shell);
}

Scene *CreateAlpha (Scene *scn) {

  Scene *alpha=NULL;
  alpha = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  SetScene(alpha,255);
  return(alpha);
}


Scene *ObjectAlpha (Scene *scn, int w) {

  int w2,imax,i;
  Curve *c;
  AdjRel3 *A;
  Scene *alpha,*dist,*border,*bin;

  w2 = w*w;
  A = Spheric(1.8);
  c = CreateCurve(2);
  c->X[0] = 0.0; 
  c->Y[0] = 0.5;
  c->X[1] = (double)ROUND((float)w2);
  c->Y[1] = 1.0;
  imax = MaximumValue3(scn);
  alpha = CreateAlpha(scn);
  for (i=1;i<=imax;i++) {
    fprintf(stderr,"Object %d: threshold...\n",i);
    bin = Threshold3(scn,i,i);
    fprintf(stderr,"Object %d: border...\n",i);
    border = InnerBorder(bin,w);
    DestroyScene(&bin);
    fprintf(stderr,"Object %d: edt...\n",i);
    dist = TDistTrans3(border,A,BOTH,w2,0);
    DestroyScene(&border);
    fprintf(stderr,"Object %d: classification...\n",i);
    ClassifyTDE(dist,alpha,scn,c,Intensity3,i);
    DestroyScene(&dist);
  }
  DestroyCurve(&c);
  DestroyAdjRel3(&A);
  return(alpha);
}

Scene *NormalIndexScene (Shell *shell) {

  Scene *scn;
  int i,i1,i2,j;
  Voxel v;

  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);
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
          j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	  scn->data[j] = shell->voxel[i].normal;
	}
      }
    }
  return(scn);
}

Scene *LabelScene (Shell *shell) {

  Scene *scn;
  int i,i1,i2,j;
  Voxel v;

  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);
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
          j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	  scn->data[j] = shell->voxel[i].obj;
	}
      }
    }
  return(scn);
}

Scene *FilledLabelScene (Shell *shell) {

  Scene *scn;
  Image *img,*fimg;
  int i;

  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);
  for (i=0;i<shell->zsize;i++) {
    img = GetShellZSlice(shell,i);
    fimg = FillSlice(img);
    DestroyImage(&img);
    PutSlice(fimg,scn,i);
    DestroyImage(&fimg);
  }
  return(scn);
}

Scene *ValueScene (Shell *shell) {

  Scene *scn;
  int i,i1,i2,j;
  Voxel v;

  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);

  if (!shell->body) return (scn);
 
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
          j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	  scn->data[j] = shell->body[i].val;	  
	}
      }
    }
  return(scn);
}

Scene *ObjectScene (Shell *shell, int obj) {

  Scene *scn;
  int i,i1,i2,j;
  Voxel v;

  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);
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
            j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	    scn->data[j] = shell->voxel[i].obj;
	  }
	}
      }
    }
  return(scn);
}

Scene *RemoveVoxels (Shell *shell, float low, float hi) {

  AdjRel3 *A;
  Scene *scn,*tmp;
  Voxel v,q;
  int i,i1,i2,j,k,n;
  uchar a_l,a_h,a;
  a_l = (uchar)low/255.;
  a_h = (uchar)hi/255.;

  A = Spheric(1.8);
  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);
  tmp = OpacityScene(shell);
  
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
	  a = shell->voxel[i].opac;
          j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
          if (a > a_l) {
            scn->data[j] = shell->voxel[i].obj;  
            for (k=0;k<A->n;k++) {
              q.x = v.x + A->dx[k];
              q.y = v.y + A->dy[k];
              q.z = v.z + A->dz[k];
 	      if (ValidVoxel(scn, q.x, q.y, q.z)) {
                n = tmp->tbz[q.z] + tmp->tby[q.y] + q.x;
                if (tmp->data[n] < a_h) {
		}
	      }
	    }
	  }
	}
      }
    }
  DestroyAdjRel3(&A);
  return(scn);
}


Shell *Scene2Shell (Scene *scn) {

  Shell *shell=NULL;
  int i,n,nvoxels;
  
  shell=(Shell *)calloc(1, sizeof(Shell));
  if (shell == NULL){
    Error(MSG1,"CreateShell");
  }

  n = scn->xsize * scn->ysize * scn->zsize; 
  
  nvoxels=0;
  for (i=0;i<n;i++)  
    if (scn->data[i] != 0) 
      nvoxels++;

  shell = NewBodyShell(scn->xsize,scn->ysize,scn->zsize,nvoxels);
  
  SetShellVoxels(shell,scn);
  SetShellList(shell,scn);
  SetBodyShellNormal(shell,scn,Gradient3);
  SetBodyShellValue(shell,scn);

  shell->dx  = scn->dx;
  shell->dy  = scn->dy;
  shell->dz  = scn->dz;
  shell->maxval = MaximumValue3(scn);
  shell->nobjs = 0;
  return shell;
}

void SetShellVoxels (Shell *shell, Scene *scn) {
  
  int i,j,k;
  Voxel v;

  i=0;
  for(v.z=0;v.z<scn->zsize;v.z++)
    for(v.y=0;v.y<scn->ysize;v.y++)
      for(v.x=0;v.x<scn->xsize;v.x++) { 
	k = scn->tbz[v.z] + scn->tby[v.y] + v.x;
        if (scn->data[k] != 0) {
	  j = shell->Myz->tbrow[v.z] + v.y;
	  if (shell->Myz->val[j] < 0) {
	    shell->Myz->val[j]=i;
	  }
  	  shell->voxel[i].x = v.x;
	  shell->voxel[i].opac = 255;
	  shell->voxel[i].visible = 1;
	  if (shell->body)
	    shell->voxel[i].obj = 0;
	  else	    
	    shell->voxel[i].obj = (uchar)scn->data[k];
	  i++;
	}
      }
}

void SetShellNormalIndex (Shell *shell, Scene *nscn) {

  int i,i1,i2,j;
  Voxel v;

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
	  j = nscn->tbz[v.z] + nscn->tby[v.y] + v.x;	  
	  shell->voxel[i].normal = (ushort)nscn->data[j];
	}        
      }
    }

}

void SetShellList (Shell *shell, Scene *scn) {

  int i,j,k;
  Voxel v;

  i=0;  
  for(v.x=0;v.x<scn->xsize;v.x++)
    for(v.z=0;v.z<scn->zsize;v.z++)
      for(v.y=0;v.y<scn->ysize;v.y++) { 
	k = scn->tbz[v.z] + scn->tby[v.y] + v.x;
        if (scn->data[k] != 0) {
	  j = shell->Mzx->tbrow[v.x] + v.z;
          if (shell->Mzx->val[j] < 0) {
            shell->Mzx->val[j]=i;
          }	  
	  shell->pointer[i].i = GetPointer(shell,&v);
	  shell->pointer[i].y = v.y;
	  i++;
	}
      }    
}


void SetShellLabel (Shell *shell, Scene *lscn) {

  int i,i1,i2,k;
  Voxel v;
  AdjRel3 *A = Spheric(1.8);

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
	  k = lscn->tbz[v.z] + lscn->tby[v.y] + v.x;	  
	  shell->voxel[i].obj = (uchar)lscn->data[k];
	}        
      }
    }
  GetShellNObjects(shell);
  DestroyAdjRel3(&A);
}

void SetShellOpacity (Shell *shell, Scene *alpha) {

  int i,i1,i2,k;
  Voxel v;

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
	  k = alpha->tbz[v.z] + alpha->tby[v.y] + v.x;	  
	  shell->voxel[i].obj = (uchar)alpha->data[k];
	}        
      }
    }

}



void SetBodyShellValue (Shell *shell, Scene *scn) {

  int i,i1,i2,k;
  Voxel v;
  AdjRel3 *A = NULL;
  AdjVxl *V = NULL;

  A = Spheric(1.8);
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
	  k = scn->tbz[v.z] + scn->tby[v.y] + v.x;	  
	  shell->body[i].val = scn->data[k];
	}        
      }
    }
  DestroyAdjRel3(&A);
  DestroyAdjVxl(&V);
}

int GetPointer(Shell *shell, Voxel *v) {
 
  int low,hi,mid,i;

  i = shell->Myz->tbrow[v->z] + v->y;

  low = shell->Myz->val[i];
  do {
    i++;
  } while (shell->Myz->val[i] < 0);
  hi = shell->Myz->val[i]-1;

  /* Binary search for index using current coord x as key*/

  while (low <= hi) {
    mid = (low + hi)/2;
    if (v->x == shell->voxel[mid].x)
      return(mid);
    if (v->x < shell->voxel[mid].x)
      hi=mid-1;
    else 
      low=mid+1;
  }
  return(-1);
}


int VoxelExist(Shell *shell, Voxel *v) {

  int i = 0;
  int j = 0;

  if (v->x < 0) return -1;
  if (v->y < 0) return -1;
  if (v->z < 0) return -1;
  if (v->x >= shell->xsize) return -1;
  if (v->y >= shell->ysize) return -1;
  if (v->z >= shell->zsize) return -1;

  i = shell->Myz->tbrow[v->z] + v->y;
  if (shell->Myz->val[i] < 0) return -1;
  j = shell->Mzx->tbrow[v->x] + v->z;
  if (shell->Mzx->val[j] < 0) return -1;
  j  = GetPointer(shell,v);
  return (j);
}


Image *InverseWarp (Image *simg, Context *cxt) {

  Image *img = CreateImage(cxt->width,cxt->height);
  int uw,vw,i,i2,iw;
  float u,v,ut,vt,w[4];
  Pixel c;

  c.x = img->nrows/2; 
  c.y = img->ncols/2;
  
  ut = cxt->isize/2;
  vt = cxt->jsize/2;

  for (vw=0;vw<img->nrows-1;vw++)
    for (uw=0;uw<img->ncols;uw++) {      

      u = (cxt->IW[0][0] * (uw-c.x) + cxt->IW[0][1] * (vw-c.y)) + ut;
      v = (cxt->IW[1][0] * (uw-c.x) + cxt->IW[1][1] * (vw-c.y)) + vt;
       
      if (ValidPixel(simg,u,v)) {

	w[0] = u - (float)(int)u;
	w[1] = v - (float)(int)v;
	w[2] = (float)((int)u + 1) - u;
	w[3] = (float)((int)v + 1) - v;

	i = simg->tbrow[(int)v] + (int)u;
	i2 = img->tbrow[(int)v+1] + (int)u;
	iw = img->tbrow[vw] + uw;

        simg->val[iw] = ((simg->val[i]*w[0] + simg->val[i+1]*w[2]) * w[1] + \
			 (simg->val[i2]*w[0] + simg->val[i2+1]*w[2]) * w[3]);

      }
    }
  return(img);
}


Image *SWShellRendering(Shell *shell,Context *cxt) {

  Image *img=NULL, *out=NULL;
  FBuffer *shear=NULL,*warp=NULL;
  ZBuffer *zbuff=NULL;
  Voxel p,q,c;
  Pixel d;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,is,iw,i,i_p,i1,i2,j=0,n;
  float k,w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,nopac[256],opac,alpha;
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  warp = CreateFBuffer(cxt->width, cxt->height);
  zbuff = CreateZBuffer(cxt->isize, cxt->jsize);
  InitFBuffer(shear,1.0);
  InitFBuffer(warp,1.0);
  InitZBuffer(cxt->zbuff);
  InitZBuffer(zbuff);
  
  img=CreateImage(cxt->isize, cxt->jsize);
  out=CreateImage(cxt->width, cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);
    
  switch(cxt->PAxis) {

  case 'x':
    
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {

	    // shear
            p.y = shell->pointer[i].y;

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;

	    u = (int) (q.y + cxt->Su[p.x]); 
	    v = (int) (q.z + cxt->Sv[p.x]); 

	    is = u + ut + shear->tbv[v + vt];

	    if (shear->val[is] > 0.05) {
	    
	      // normal
	      i_p = shell->pointer[i].i;
	      l = shell->voxel[i_p].obj;
	      
	      if (l) {
		n = shell->voxel[i_p].normal;
		
		cos_a = cxt->viewer.x * shell->normaltable[n].x +\
		  cxt->viewer.y * shell->normaltable[n].y +\
		  cxt->viewer.z * shell->normaltable[n].z;

		if (cos_a > 0.0) {

		  // shading

		  w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		  k = cxt->depth - w;
		  idist = 1.0 - w/(float)cxt->depth;

		  cos_2a = 2*cos_a*cos_a - 1.0;
	      
		  if (cos_2a < 0.) {
		    pow = 0.;
		    spec = 0.;
		  } else {
		    pow = 1.;
		    for (j=0; j < cxt->obj[l].ns; j++) 
		      pow = pow * cos_2a;
		    spec = cxt->obj[l].spec * pow;
		  }

		  opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		  amb  = cxt->obj[l].amb;
		  diff = cxt->obj[l].diff * cos_a;

		  shading = (amb + idist * (diff + spec));
		  alpha = 255. * opac * shading;
		  shear->val[is] *= (1.0-opac);
		  if (shear->val[is] > .05) {
		    img->val[is] += (int)(alpha * shear->val[is] * cxt->footprint->val[j]);
		    if (zbuff->voxel[is] == NIL) {
		      zbuff->voxel[is] = i_p;
		      zbuff->object[is] = l;
		      if (zbuff->dist[is] > k)
			zbuff->dist[is] = k;
		    }
		  }
		}
	      }
	    }	    
	  }
	}
      }

    break;
    
  case 'y': 
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;
	   
	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    
	    u = (int) (q.z + cxt->Su[p.y]); 
	    v = (int) (q.x + cxt->Sv[p.y]); 

	    is = u + ut+ shear->tbv[v + vt];

	    if (shear->val[is] > 0.05) {
	      
	      // normal
	      i_p = i;
	      l = shell->voxel[i_p].obj;
	      if (l) {
		n = shell->voxel[i_p].normal;

		cos_a = cxt->viewer.x * shell->normaltable[n].x +\
		  cxt->viewer.y * shell->normaltable[n].y +\
		  cxt->viewer.z * shell->normaltable[n].z;

		if (cos_a > 0.0) {

		  // shading

		  w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		  k = cxt->depth - w;
		  idist = 1.0 - w/(float)cxt->depth;

		  cos_2a = 2*cos_a*cos_a - 1.0;
		
		  if (cos_2a < 0.) {
		    pow = 0.;
		    spec = 0.;
		  } else {
		    pow = 1.;
		    for (j=0; j < cxt->obj[l].ns; j++) 
		      pow = pow * cos_2a;
		    spec = cxt->obj[l].spec * pow;
		  }

		  opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		  amb  = cxt->obj[l].amb;
		  diff = cxt->obj[l].diff * cos_a;
		
		  shading = (float)255. * opac * (amb + idist * (diff + spec));
		  alpha = 255. * opac * shading;
		  shear->val[is] *= (1.0-opac);
		  if (shear->val[is] > .05) {
		    img->val[is] += (int)(alpha * shear->val[is] * cxt->footprint->val[j]);		  		    
		    if (zbuff->voxel[is] == NIL) {
		      zbuff->voxel[is] = i_p;
		      zbuff->object[is] = l;
		      if (zbuff->dist[is] > k)
			zbuff->dist[is] = k;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;

  case 'z': 
    
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z; 

	    u = (int) (q.x + cxt->Su[p.z]); 
	    v = (int) (q.y + cxt->Sv[p.z]); 

	    is = u + ut + shear->tbv[v + vt];
	    
	    if (shear->val[is] > 0.05) {
	      
	      // normal
	      i_p = i;
	      l = shell->voxel[i_p].obj;
	      if (l) {

		n = shell->voxel[i_p].normal;

		cos_a = cxt->viewer.x * shell->normaltable[n].x +\
		  cxt->viewer.y * shell->normaltable[n].y +\
		  cxt->viewer.z * shell->normaltable[n].z;

		if (cos_a > 0.0) {

                    	       		  
		  // shading
		  
		  w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		  k = cxt->depth - w;
		  idist = 1.0 - w/(float)cxt->depth;
		  
		  cos_2a = 2*cos_a*cos_a - 1.0;
		  
		  if (cos_2a < 0.) {
		    pow = 0.;
		    spec = 0.;
		  } else {
		    pow = 1.;
		    for (j=0; j < cxt->obj[l].ns; j++) 
		      pow = pow * cos_2a;
		    spec = cxt->obj[l].spec * pow;
		  }
		  
		  opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		  amb  = cxt->obj[l].amb;
		  diff = cxt->obj[l].diff * cos_a;
		  
		  shading = (amb + idist * (diff + spec));
		  alpha = 255. * opac * shading;
		  shear->val[is] *= (1.0-opac);
		  if (shear->val[is] > .05) {
		    img->val[is] += (int)(alpha * shear->val[is]);	  
		    if (zbuff->voxel[is] == NIL) {
		      zbuff->voxel[is] = i_p;
		      zbuff->object[is] = l;
		      if (zbuff->dist[is] > k)
			zbuff->dist[is] = k;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;
    
  default: 
    Error(MSG1,"SWShellRendering");
    break;    
  }

  for (v =0;v<cxt->height;v++)
    for (u =0;u<cxt->width;u++) {
      uw = u - d.x;
      vw = v - d.y;
      p.x = cxt->IW[0][0] * uw + cxt->IW[0][1] * vw + ut;
      p.y = cxt->IW[1][0] * uw + cxt->IW[1][1] * vw + vt;
      if (ValidPixel(img,p.x,p.y)) {
	iw = u + cxt->zbuff->tbv[v];
	is = p.x + zbuff->tbv[p.y];
	out->val[iw] = img->val[is];
	cxt->zbuff->object[iw] = zbuff->object[is];
	cxt->zbuff->voxel[iw] = zbuff->voxel[is];
	cxt->zbuff->dist[iw] = zbuff->dist[is];
      }
    }

  DestroyImage(&img);
  DestroyFBuffer(&shear);
  DestroyFBuffer(&warp);
  DestroyZBuffer(&zbuff);			
  DestroyAdjPxl(&fprint);
  return(out);
}
		
Image *ShellRendering (Shell *shell, Context *cxt) {

  Image *img=NULL;
  FBuffer *buff=NULL;
  Voxel c,p,q;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,i1,i2,i,i_p,is,j,n,ut,vt;
  float k,w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,nopac[256],opac,alpha;
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // image translation
  ut = cxt->width/2; 
  vt = cxt->height/2;
  wt = (float)cxt->depth/2;

  buff = CreateFBuffer(cxt->width, cxt->height);
  InitFBuffer(buff,1.0);
 
  img = CreateImage(cxt->width, cxt->height);
  fprint = AdjPixels(img,cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;

	for (i = i1; i != i2; i += cxt->dx) {
	  l = shell->voxel[i].obj;
	  if (l) {
	    p.x = shell->voxel[i].x;

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	  
	    u = (int)((cxt->R[0][0]*q.x) + (cxt->R[0][1]*q.y) + (cxt->R[0][2]*q.z) + ut);
	    v = (int)((cxt->R[1][0]*q.x) + (cxt->R[1][1]*q.y) + (cxt->R[1][2]*q.z) + vt);

	    i_p = u + buff->tbv[v];
	  
	    if (buff->val[i_p] > 0.05){

	      // normal
	      n = shell->voxel[i].normal;
	    
	      cos_a = viewer.x * shell->normaltable[n].x +\
		viewer.y * shell->normaltable[n].y +\
		viewer.z * shell->normaltable[n].z;
	    
	      if (cos_a > 0) {
	      
		// shading
		w = (cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z)+ wt;

		k = cxt->depth - w;
		idist = 1.0 - (w / (float)cxt->depth);

		cos_2a = 2*cos_a*cos_a - 1.0;
	      
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}
		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;
	      
		shading =  (amb + idist * (diff + spec));
		alpha = 255.* opac * shading;
		
		/* Splatting */
		for (j=0; j < fprint->n; j++){		
		  is = i_p + fprint->dp[j];
		 
		  if (buff->val[is] > .05) {                
		    img->val[is] += (int)(alpha * buff->val[is] * cxt->footprint->val[j]);
		    buff->val[is] *= (1.0- (opac * cxt->footprint->val[j]));
		    if (cxt->zbuff->voxel[is] == NIL) {
		      cxt->zbuff->voxel[is] = i_p;
		      cxt->zbuff->object[is] = l;
		      if (cxt->zbuff->dist[is] > k)
			cxt->zbuff->dist[is] = k;
		    }
		  }
		}	
	      }  
	    }
	  }
	}
      }
    }

  DestroyFBuffer(&buff);
  DestroyAdjPxl(&fprint);
  
  return(img);
}

CImage *CShellRendering (Shell *shell, Context *cxt) {

  CImage *cimg=NULL;
  FBuffer *buff=NULL;
  Voxel c,p,q;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,i1,i2,i,i_p,is,j,n,ut,vt,k;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,nopac[256],opac,alpha;
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // image translation
  ut = cxt->width/2; 
  vt = cxt->height/2;
  wt = (float)cxt->depth/2;

  buff = CreateFBuffer(cxt->width, cxt->height);
  InitFBuffer(buff,1.0);
 
  cimg = CreateCImage(cxt->width, cxt->height);
  fprint = AdjPixels(cimg->C[0],cxt->footprint->adj);

  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];
    
  for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
      i = shell->Myz->tbrow[p.z] + p.y;
        
      if (shell->Myz->val[i] >= 0) {
	
	if (cxt->dx == 1) {              
	  i1 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i2 = shell->Myz->val[i]-1;            
	} else {            
	  i2 = shell->Myz->val[i];
	  i++;
	  while (shell->Myz->val[i] < 0) {
	    i++;
	  } 
	  i1 = shell->Myz->val[i]-1;
	}
	i2 += cxt->dx;

	for (i = i1; i != i2; i += cxt->dx) {
	  l = shell->voxel[i].obj;
	  if (l) {
	    p.x = shell->voxel[i].x;

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	  
	    u = (int)((cxt->R[0][0]*q.x) + (cxt->R[0][1]*q.y) + (cxt->R[0][2]*q.z) + ut);
	    v = (int)((cxt->R[1][0]*q.x) + (cxt->R[1][1]*q.y) + (cxt->R[1][2]*q.z) + vt);

	    i_p = u + buff->tbv[v];
	  
	    if (buff->val[i_p] > 0.05){

	      // normal
	      n = shell->voxel[i].normal;
	    
	      cos_a = viewer.x * shell->normaltable[n].x +\
		viewer.y * shell->normaltable[n].y +\
		viewer.z * shell->normaltable[n].z;
	    
	      if (cos_a > 0) {
	      
		// shading
		w = (cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z)+ wt;
		k = cxt->depth - w;
		idist = 1.0 - (w / (float)cxt->depth);

		cos_2a = 2*cos_a*cos_a - 1.0;
	      
		if (cos_2a < 0.) {
		  pow = 0.;
		  spec = 0.;
		} else {
		  pow = 1.;
		  for (j=0; j < cxt->obj[l].ns; j++) 
		    pow = pow * cos_2a;
		  spec = cxt->obj[l].spec * pow;
		}
		opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		amb  = cxt->obj[l].amb;
		diff = cxt->obj[l].diff * cos_a;
	     
		shading =  (amb + idist * (diff + spec));
		/* Splatting */
		for (j=0; j < fprint->n; j++){		
		  is = i_p + fprint->dp[j];
		  if (buff->val[is] > .05) {
		    alpha = cxt->footprint->val[j] * opac;
		    AccVoxelColor(cimg,cxt,is,shading,buff->val[is]*alpha,l);
		    buff->val[is] *= (1.0-alpha);
		    if (cxt->zbuff->voxel[i_p] == NIL) {
		      cxt->zbuff->voxel[i_p] = i_p;
		      cxt->zbuff->object[i_p] = l;		      
		      if (cxt->zbuff->dist[i_p] > k)
			cxt->zbuff->dist[i_p] = k;
		    }
		  }
		}	
	      }  
	    }
	  }
	}
      }
    }

  DestroyFBuffer(&buff);
  DestroyAdjPxl(&fprint);
  
  return(cimg);
}





CImage *CMIPShell(Shell *shell, Context *cxt) {

  CImage *cimg=NULL;
  Image *img=NULL;

  img = MIPShell(shell,cxt);
  cimg = Colorize(cxt,img);
  DestroyImage(&img);
  return(cimg);
}
  

CImage *CSWShellRendering(Shell *shell,Context *cxt) {


  CImage *cimg=NULL;
  CImage *out=NULL; 
  FBuffer *shear=NULL;
  ZBuffer *zbuff=NULL;
  Voxel p,q,c;
  Pixel d,min,max;
  Vector viewer;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,uw,vw,is,iw,i,i_p,i1,i2,j,n;
  float w,wt,amb,spec,diff,shading,idist,cos_a,cos_2a,pow,alpha,opac,nopac[256],k; 
  uchar l;

  for(i=0;i<256;i++)
    nopac[i] = (float)i/255.;

  // image translation
  ut = cxt->isize/2; 
  vt = cxt->jsize/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  // warp buffer translation
  d.x = cxt->width/2;
  d.y = cxt->height/2;

  shear = CreateFBuffer(cxt->isize, cxt->jsize);
  zbuff = CreateZBuffer(cxt->isize, cxt->jsize);
  InitFBuffer(shear,1.0);
  InitZBuffer(cxt->zbuff);
  InitZBuffer(zbuff);

  out=CreateCImage(cxt->width, cxt->height);
  cimg =CreateCImage(cxt->isize, cxt->jsize);
  fprint = AdjPixels(cimg->C[0],cxt->footprint->adj);
    
  viewer.x = -cxt->IR[0][2];
  viewer.y = -cxt->IR[1][2];
  viewer.z = -cxt->IR[2][2];

  max.x = max.y = INT_MIN;
  min.x = min.y = INT_MAX;
    
  switch(cxt->PAxis) {

  case 'x':

    for (p.x = cxt->ki; p.x != cxt->kf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {
	    // shear
            p.y = shell->pointer[i].y;

	    if (cxt->ymin <= p.y && p.y <= cxt->ymax) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	      
	      u = ROUND (q.y + cxt->Su[p.x]); 
	      v = ROUND (q.z + cxt->Sv[p.x]); 
	    
	      is = u + ut + shear->tbv[v + vt];
	    
	      if (shear->val[is] > 0.05) {
		
		// normal
		i_p = shell->pointer[i].i;
		
		if (shell->voxel[i_p].visible) {		
		
		  l = shell->voxel[i_p].obj;
		  if (l) {

		    n = shell->voxel[i_p].normal;	      
		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;
		  
		    if (cos_a > 0 || shell->body) {
		    
		      // shading
		    
		      w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
                      k = cxt->depth - w;
		      idist = 1.0 - w/(float)cxt->depth;
		    
		      cos_2a = 2*cos_a*cos_a - 1.0;
		    
		      if (cos_2a < 0.) {
			pow = 0.;
			spec = 0.;
		      } else {
			pow = 1.;
			for (j=0; j < cxt->obj[l].ns; j++) 
			  pow = pow * cos_2a;
			spec = cxt->obj[l].spec * pow;
		      }
		    
		      opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		      amb  = cxt->obj[l].amb;
		      diff = cxt->obj[l].diff * cos_a;
		    
		      shading = (amb + idist * (diff + spec));		  
		      alpha = opac;
		      AccVoxelColor(cimg,cxt,is,shading,shear->val[is]*alpha,l);
		      shear->val[is] *= (1.0-alpha);
		      if (zbuff->voxel[is] == NIL) {
			zbuff->voxel[is] = i_p;
			zbuff->object[is] = l;
			if (zbuff->dist[is] > k)
			  zbuff->dist[is] = k;
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;
    
  case 'y': 
    
    for (p.y = cxt->ki; p.y != cxt->kf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;

	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z;
	    
	      u = ROUND (q.z + cxt->Su[p.y]); 
	      v = ROUND (q.x + cxt->Sv[p.y]); 

	      is = u + ut+ shear->tbv[v + vt];

	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {
		  l = shell->voxel[i_p].obj;
		  if (l) {
		    n = shell->voxel[i_p].normal;

		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;

		    if (cos_a > 0  || shell->body) {

		      // shading

		      w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
                      k = cxt->depth - w;
		      idist = 1.0 - w/(float)cxt->depth;

		      cos_2a = 2*cos_a*cos_a - 1.0;
		
		      if (cos_2a < 0.) {
			pow = 0.;
			spec = 0.;
		      } else {
			pow = 1.;
			for (j=0; j < cxt->obj[l].ns; j++) 
			  pow = pow * cos_2a;
			spec = cxt->obj[l].spec * pow;
		      }

		      opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac;
		      amb  = cxt->obj[l].amb;
		      diff = cxt->obj[l].diff * cos_a;
		
		      shading =  (amb + idist * (diff + spec));
		
		      // splatting

		      alpha = opac;
		      AccVoxelColor(cimg,cxt,is,shading,shear->val[is]*alpha,l);		      
		      shear->val[is] *= (1.0-alpha);
		      if (zbuff->voxel[is] == NIL) {
			zbuff->voxel[is] = i_p;
			zbuff->object[is] = l;
			if (zbuff->dist[is] > k)
			  zbuff->dist[is] = k;			    
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;

  case 'z': 

    for (p.z = cxt->ki; p.z != cxt->kf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // shear
            p.x = shell->voxel[i].x;

	    if (cxt->xmin <= p.x && p.x <= cxt->xmax) {

	      q.x = p.x - c.x;
	      q.y = p.y - c.y;
	      q.z = p.z - c.z; 

	      u = ROUND (q.x + cxt->Su[p.z]); 
	      v = ROUND (q.y + cxt->Sv[p.z]); 

	      
	      is = u + ut + shear->tbv[v + vt];
	      
	      if (shear->val[is] > 0.05) {
	      
		// normal
		i_p = i;

		if (shell->voxel[i_p].visible) {
		  l = shell->voxel[i_p].obj;
		  if (l) {
		    n = shell->voxel[i_p].normal;

		    cos_a = viewer.x * shell->normaltable[n].x +\
		      viewer.y * shell->normaltable[n].y +\
		      viewer.z * shell->normaltable[n].z;

		    if (cos_a > 0  || shell->body) {

		      // shading

		      w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		      k = cxt->depth - w;
		      idist = 1.0 - w/(float)cxt->depth;

		      cos_2a = 2*cos_a*cos_a - 1.0;
		
		      if (cos_2a < 0.) {
			pow = 0.;
			spec = 0.;
		      } else {
			pow = 1.;
			for (j=0; j < cxt->obj[l].ns; j++) 
			  pow = pow * cos_2a;
			spec = cxt->obj[l].spec * pow;
		      }
		
		      opac = nopac[shell->voxel[i_p].opac] * cxt->obj[l].opac; 
		      amb  = cxt->obj[l].amb;
		      diff = cxt->obj[l].diff * cos_a;
		
		      shading =  (amb + idist * (diff + spec));
		
		      alpha = opac;
		      AccVoxelColor(cimg,cxt,is,shading,shear->val[is]*alpha,l);
		      shear->val[is] *= (1.0-alpha);
		      if (zbuff->voxel[is] == NIL) {
			zbuff->voxel[is] = i_p;
			zbuff->object[is] = l;
			if (zbuff->dist[is] > k)
			  zbuff->dist[is] = k;
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    break;    
      
  default: 
    Error(MSG1,"SWShellRendering");
    break;
    
  }
  for (v =0;v<cxt->height;v++)
    for (u =0;u<cxt->width;u++) {
      uw = u - d.x;
      vw = v - d.y;
      p.x = cxt->IW[0][0] * uw + cxt->IW[0][1] * vw + ut;
      p.y = cxt->IW[1][0] * uw + cxt->IW[1][1] * vw + vt;
      if (ValidPixel(cimg->C[0],p.x,p.y)) {
	iw = u + cxt->zbuff->tbv[v];
	is = p.x + zbuff->tbv[p.y];
	out->C[0]->val[iw] = cimg->C[0]->val[is];
	out->C[1]->val[iw] = cimg->C[1]->val[is];
	out->C[2]->val[iw] = cimg->C[2]->val[is];
	cxt->zbuff->object[iw] = zbuff->object[is];
	cxt->zbuff->voxel[iw] = zbuff->voxel[is];
	cxt->zbuff->dist[iw] = zbuff->dist[is];
      }
    }
  DestroyCImage(&cimg);
  DestroyFBuffer(&shear);
  DestroyZBuffer(&zbuff);
  DestroyAdjPxl(&fprint);
  return(out);
}





CImage *CSWShellRenderingStereo(Shell *shell,Context *cxt) {

  CImage *cimg = NULL;
  Image *r=NULL;
  Image *b=NULL;
  Image *g=NULL;
  
  cimg=(CImage*)calloc(1,sizeof(CImage));

  AddAngles(cxt,0.,-5.);
  r = SWShellRendering(shell,cxt);
  AddAngles(cxt,0.,10.);
  b = SWShellRendering(shell,cxt);
  AddAngles(cxt,0.,-5.);
  g = SWShellRendering(shell,cxt);

  cimg->C[1] = g;
  cimg->C[0] = r;//Or(r,cimg->C[1]);
  cimg->C[2] = b;

  //  DestroyImage(&r);
  //DestroyImage(&b); 
  return(cimg);
}

CImage *CShellRenderingStereo(Shell *shell,Context *cxt) {

  CImage *cimg = NULL;
  Image *r=NULL;
  Image *b=NULL;
  Image *g=NULL;
  
  cimg=(CImage*)calloc(1,sizeof(CImage));

  AddAngles(cxt,0.,-5.);
  r = ShellRendering(shell,cxt);
  AddAngles(cxt,0.,10.);
  b = ShellRendering(shell,cxt);
  AddAngles(cxt,0.,-5.);
  g = ShellRendering(shell,cxt);

  cimg->C[1] = g;
  cimg->C[0] = r;//Or(r,cimg->C[1]);
  cimg->C[2] = b;

  //  DestroyImage(&r);
  //DestroyImage(&b); 
  return(cimg);
}



int GetShellXSize (Shell *shell) {
  return (shell->xsize);
}
int GetShellYSize (Shell *shell) {
  return (shell->ysize);
}
int GetShellZSize (Shell *shell) {
  return (shell->zsize);
}



int GetShellMaximum (Shell *shell) {

  Voxel v;
  int i,i1,i2;
  int imax=0;

  if (shell->scn) return (MaximumValue3(shell->scn));
  if (!shell->body) return 0;

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
	  if (shell->voxel[i].visible)
	    imax = MAX(imax,shell->body[i].val);	  
	}        
      }
    }
  shell->maxval = imax;
  return(imax);
}

int GetShellMinimum (Shell *shell) {

  Voxel v;
  int i,i1,i2;
  int imin=INT_MAX;

  if (shell->scn) return (MinimumValue3(shell->scn));
  if (!shell->body) return 0;

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
	  if (shell->voxel[i].visible)
	    imin = MIN(imin,shell->body[i].val);
	}        
      }
    }
  return(imin);
}

int GetShellNObjects(Shell *shell) {

  int i;
  int buf[256];
  int n=0;

  for(i=0;i<256;i++) {
    buf[i] = 0;
  }
  for(i=0;i<shell->nvoxels;i++) {
    buf[shell->voxel[i].obj]++;
  }
  for(i=1;i<256;i++) {
    if(buf[i])
      n++;
  }
  shell->nobjs = (uchar)n;
  return n;
}


Curve *ShellObjHistogram(Shell *shell)
{
  int i,p,n,nbins;
  Curve *hist=NULL;
  nbins = 256;
  hist  = CreateCurve(nbins);
  n     = shell->nvoxels;
  
  for (p=0; p < n; p++) {
    hist->Y[shell->voxel[p].obj]++;
  }
  for (i=0; i < nbins; i++) 
    hist->X[i] = i;
  
  return(hist);
}


Curve *ShellHistogram(Shell *shell)
{
  int i,p,n,nbins;
  Curve *hist=NULL;
  
  if (!shell->body) {
    hist = ShellObjHistogram(shell);
    return(hist);
  }
  
  nbins = GetShellMaximum (shell) + 1;
  hist  = CreateCurve(nbins);
  n     = shell->nvoxels;
  
  for (p=0; p < n; p++) {
    hist->Y[shell->body[p].val]++;
  }
  for (i=0; i < nbins; i++) 
    hist->X[i] = i;

  return(hist);
}

Curve *ShellNormHistogram(Shell *shell) 
{
  int i,sum;
  Curve *hist=NULL,*nhist=NULL;

  hist  = ShellHistogram(shell);
  if (shell->body)
    sum   = shell->nvoxels;
  else 
    sum   = shell->nobjs;
  nhist = CreateCurve(hist->n);
  for (i=0; i < nhist->n;i++){
    nhist->Y[i] = hist->Y[i]/sum;
    nhist->X[i] = hist->X[i];
  }

  DestroyCurve(&hist);

  return(nhist);
}

Curve *ShellGradientHistogram(Shell *shell)
{
  int i,p,n,nbins;
  Curve *hist=NULL;
  
  if (!shell->body) {
    hist = ShellObjHistogram(shell);
    return(hist);
  }
  
  nbins = GetShellMaximum (shell) + 1;
  hist  = CreateCurve(nbins);
  n     = shell->nvoxels;
  
  for (p=0; p < n; p++) {
    hist->Y[shell->body[p].normalmag]++;
  }
  for (i=0; i < nbins; i++) 
    hist->X[i] = i;

  return(hist);
}

Image *GetShellXSlice (Shell *shell, int x) {

  Voxel v;
  int i,i1,i2,j;
  Image *img= NULL;

  img = CreateImage(shell->ysize,shell->zsize);
  v.x = x;
  for (v.z = 0; v.z != shell->zsize; v.z ++) {
    i = shell->Mzx->tbrow[v.x] + v.z;     
    if (shell->Mzx->val[i] >= 0) {	
      i1 = shell->Mzx->val[i];
      i++;
      while (shell->Mzx->val[i] < 0) {
	i++;
      } 
      i2 = shell->Mzx->val[i];
      for (i = i1; i != i2; i++) {
	v.y = shell->pointer[i].y;
	j = img->tbrow[v.z] + v.y;
	if (shell->voxel[shell->pointer[i].i].visible) {
	  img->val[j] = shell->voxel[shell->pointer[i].i].obj;
	} else {
	  img->val[j] = 256; // > 255 = invisible
	}
      }        
    }
  }
  return(img);
}


Image *GetShellYSlice (Shell *shell, int y) {

  Voxel v;
  int i,i1,i2,j;
  Image *img= NULL;

  img = CreateImage(shell->xsize,shell->zsize);
  v.y = y;
  for (v.z = 0; v.z != shell->zsize; v.z ++) {
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
	j = img->tbrow[v.z] + v.x;	  
	if (shell->voxel[i].visible) {
	  img->val[j] = shell->voxel[i].obj;
	} else {
	  img->val[j] = 256; // > 255 = invisible
	}
      }        
    }
  }
  return(img);
}

Image *GetShellZSlice (Shell *shell, int z) {

  Voxel v;
  int i,i1,i2,j;
  Image *img= NULL;

  img = CreateImage(shell->xsize,shell->ysize);

  v.z = z;
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
	j = img->tbrow[v.y] + v.x;
	if (shell->voxel[i].visible) {
	  img->val[j] = shell->voxel[i].obj;
	} else {
	  img->val[j] = 256; // > 255 = invisible
	}
      }        
    }
  }
  return(img);
}


Image *GetBodyShellXSlice (Shell *shell, int x) {

  Voxel v;
  int i,i1,i2,j;
  Image *img= NULL;

  img = CreateImage(shell->ysize,shell->zsize);
  if (shell->body != NULL) {
    v.x = x;
    for (v.z = 0; v.z != shell->zsize; v.z ++) {
      i = shell->Mzx->tbrow[v.x] + v.z;     
      if (shell->Mzx->val[i] >= 0) {	
	i1 = shell->Mzx->val[i];
	i++;
	while (shell->Mzx->val[i] < 0) {
	  i++;
	} 
	i2 = shell->Mzx->val[i];
	for (i = i1; i != i2; i++) {
	  //	  if (shell->voxel[shell->pointer[i].i].visible) {
	  v.y = shell->pointer[i].y;
	  j = img->tbrow[v.z] + v.y;
	  img->val[j] = shell->body[shell->pointer[i].i].val * shell->voxel[shell->pointer[i].i].opac/255.;
	  //	  }
	}
      }
    }
  }
  return(img);
}


Image *GetBodyShellYSlice (Shell *shell, int y) {

  Voxel v;
  int i,i1,i2,j;
  Image *img= NULL;

  img = CreateImage(shell->xsize,shell->zsize);
  if (shell->body != NULL) {
    v.y = y;
    for (v.z = 0; v.z != shell->zsize; v.z ++) {
      i = shell->Myz->tbrow[v.z] + v.y;     
      if (shell->Myz->val[i] >= 0) {	
	i1 = shell->Myz->val[i];
	i++;
	while (shell->Myz->val[i] < 0) {
	  i++;
	} 
	i2 = shell->Myz->val[i];
	for (i = i1; i != i2; i++) {	  
	  //if (shell->voxel[i].visible) {
	  v.x = shell->voxel[i].x;
	  j = img->tbrow[v.z] + v.x;
	  img->val[j] = shell->body[i].val * shell->voxel[i].opac/255.;
	  //}
	}        
      }
    }
  }
  return(img);
}

Image *GetBodyShellZSlice (Shell *shell, int z) {

  Voxel v;
  int i,i1,i2,j;
  Image *img= NULL;

  img = CreateImage(shell->xsize,shell->ysize);
  if (shell->body != NULL) {
    v.z = z;
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
	  //if (shell->voxel[i].visible) {
	  v.x = shell->voxel[i].x;
	  j = img->tbrow[v.y] + v.x;
	  img->val[j] = shell->body[i].val * shell->voxel[i].opac/255.;
	  //}
	}        
      }
    }
  }
  return(img);
}

void SetObjectVisibility (Shell *shell, int obj, int visible)    
{

  Voxel v;
  int i,i1,i2;

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
	  if (shell->voxel[i].obj == obj)
	    shell->voxel[i].visible = visible; 
	}        
      }
    }
}

void SetBodyShellVisibility (Shell *shell, int lower, int higher)    
{

  Voxel v;
  int i,i1,i2;
  
  if (shell->body)
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
	    if (shell->body[i].val >= lower && shell->body[i].val <= higher)
	      shell->voxel[i].visible = 1;
	    else 
	      shell->voxel[i].visible = 0;            
	  }        
	}
      }
}


Image *MIPShell(Shell *shell, Context *cxt) {

  Image *img=NULL;
  Voxel p,q,c;
  AdjPxl *fprint=NULL;

  int u,v,ut,vt,iw,iw_p,i,i_p,i1,i2,j,val;
  float w,wt,k;
  uchar l;


  // image translation
  ut = cxt->width/2; 
  vt = cxt->height/2;
  wt = (float)cxt->depth/2;

  // scene translation
  c.x = cxt->xsize/2;
  c.y = cxt->ysize/2;
  c.z = cxt->zsize/2;

  img=CreateImage(cxt->width, cxt->height);
  
  if (!shell->body) return(img);

  InitZBuffer(cxt->zbuff);
  
  fprint = AdjPixels(img,cxt->footprint->adj);

  switch(cxt->PAxis) {

  case 'x':
    
    for (p.x = cxt->xi; p.x != cxt->xf; p.x += cxt->dx) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Mzx->tbrow[p.x] + p.z;
        
        if (shell->Mzx->val[i] >= 0) {
          
          if (cxt->dy == 1) {              
            i1 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i2 = shell->Mzx->val[i]-1;            
          } else {            
            i2 = shell->Mzx->val[i];
            i++;
            while (shell->Mzx->val[i] < 0) {
              i++;
            } 
            i1 = shell->Mzx->val[i]-1;
	  }
	  i2 += cxt->dy;          
          for (i = i1; i != i2; i += cxt->dy) {
	    
	    // projection
            p.y = shell->pointer[i].y;
	    i_p = shell->pointer[i].i;

	    if (shell->voxel[i_p].visible) {

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    
	    u = (int)((cxt->R[0][0]*q.x) + (cxt->R[0][1]*q.y) + (cxt->R[0][2]*q.z) + ut);
	    v = (int)((cxt->R[1][0]*q.x) + (cxt->R[1][1]*q.y) + (cxt->R[1][2]*q.z) + vt);
	    
	    iw = u + img->tbrow[v];

	    // splatting
	    for (j=0; j < fprint->n; j++){		
	      iw_p = iw + fprint->dp[j];
	      val = shell->body[i_p].val * cxt->footprint->val[j];
	      if (val > img->val[iw_p]) {
		img->val[iw_p] = val;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		l = shell->voxel[i_p].obj;
                k = cxt->depth - w;
		cxt->zbuff->voxel[iw_p] = i_p;
		cxt->zbuff->object[iw_p] = l;
		if (cxt->zbuff->dist[iw_p] > k)
		  cxt->zbuff->dist[iw_p] = k;
	      }
	    }
	    }
	  }
	}
      }
    break;
    
  case 'y': 
    
    for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) 
      for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {

	    // projection
            p.x = shell->voxel[i].x;
	    i_p = i;

	    if (shell->voxel[i_p].visible) {

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    
	    u = (int)((cxt->R[0][0]*q.x) + (cxt->R[0][1]*q.y) + (cxt->R[0][2]*q.z) + ut);
	    v = (int)((cxt->R[1][0]*q.x) + (cxt->R[1][1]*q.y) + (cxt->R[1][2]*q.z) + vt);
	    
	    iw = u + img->tbrow[v];
	    // splatting
	    for (j=0; j < fprint->n; j++){		
	      iw_p = iw + fprint->dp[j];
	      val = shell->body[i_p].val * cxt->footprint->val[j];
	      if (val > img->val[iw_p]) {
		img->val[iw_p] = val;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		l = shell->voxel[i_p].obj;
                k = cxt->depth - w;
		cxt->zbuff->voxel[iw_p] = i_p;
		cxt->zbuff->object[iw_p] = l;
		if (cxt->zbuff->dist[iw_p] > k)
		  cxt->zbuff->dist[iw_p] = k;
	      }
	    }
	    }
	  }
	}
      }

    break;

  case 'z': 
    
    for (p.z = cxt->zi; p.z != cxt->zf; p.z += cxt->dz) 
      for (p.y = cxt->yi; p.y != cxt->yf; p.y += cxt->dy) {
        
        i = shell->Myz->tbrow[p.z] + p.y;
        
        if (shell->Myz->val[i] >= 0) {
          
          if (cxt->dx == 1) {              
            i1 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i2 = shell->Myz->val[i]-1;            
          } else {            
            i2 = shell->Myz->val[i];
            i++;
            while (shell->Myz->val[i] < 0) {
              i++;
            } 
            i1 = shell->Myz->val[i]-1;
	  }
	  i2 += cxt->dx;          
          for (i = i1; i != i2; i += cxt->dx) {
	    // projection
            p.x = shell->voxel[i].x;
	    i_p = i;

	    if (shell->voxel[i_p].visible) {

	    q.x = p.x - c.x;
	    q.y = p.y - c.y;
	    q.z = p.z - c.z;
	    
	    u = (int)((cxt->R[0][0]*q.x) + (cxt->R[0][1]*q.y) + (cxt->R[0][2]*q.z) + ut);
	    v = (int)((cxt->R[1][0]*q.x) + (cxt->R[1][1]*q.y) + (cxt->R[1][2]*q.z) + vt);
	    
	    iw = u + img->tbrow[v];
	    // splatting	    
	    for (j=0; j < fprint->n; j++){		
	      iw_p = iw + fprint->dp[j];
	      val = shell->body[i_p].val * cxt->footprint->val[j];
	      if (val > img->val[iw_p]) {
		img->val[iw_p] = val;
		w = cxt->R[2][0]*q.x + cxt->R[2][1]*q.y + cxt->R[2][2]*q.z + wt;
		l = shell->voxel[i_p].obj;
                k = cxt->depth - w;
		cxt->zbuff->voxel[iw_p] = i_p;
		cxt->zbuff->object[iw_p] = l;
		if (cxt->zbuff->dist[iw_p] > k)
		  cxt->zbuff->dist[iw_p] = k;
	      }	    
	    }
	    }
	  }	  
	}
      }
    break;
      
  default: 
    Error(MSG1,"MIPShell");
    break;
    
  }			
  DestroyAdjPxl(&fprint);

  return(img);
}


Scene *OpacityScene (Shell *shell) {

  Scene *scn;
  int i,i1,i2,j;
  Voxel v;

  scn = CreateScene(shell->xsize,shell->ysize,shell->zsize);
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
          j = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	  scn->data[j] = shell->voxel[i].opac;
	}
      }
    }
  return(scn);
}

