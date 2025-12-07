#include "mathematics3.h"
#include "common.h"

Scene *Diff3(Scene *scn1, Scene *scn2)
{
  Scene *diff;
  int p, n;

  diff = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for (p = 0; p < n; p++)
    diff->data[p] = MAX(scn1->data[p] - scn2->data[p], 0);

  return (diff);
}

Scene *Sum3(Scene *scn1, Scene *scn2)
{
  Scene *sum;
  int p, n;

  sum = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for (p = 0; p < n; p++)
    sum->data[p] = scn1->data[p] + scn2->data[p];

  return (sum);
}

Scene *Or3(Scene *scn1, Scene *scn2)
{
  Scene *or;
  int p, n;

  or = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for (p = 0; p < n; p++)
    or->data[p] = MAX(scn1->data[p],scn2->data[p]);

  return (or);
}

Scene *And3(Scene *scn1, Scene *scn2)
{
  Scene *and;
  int p, n;

  and = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for (p = 0; p < n; p++)
    and->data[p] = MIN(scn1->data[p],scn2->data[p]);

  return (and);
}


Scene *XOr3(Scene *scn1, Scene *scn2){
  Scene *xor;
  int p, n;

  xor = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for (p = 0; p < n; p++)
    xor->data[p] = MAX(scn1->data[p],scn2->data[p])-
                   MIN(scn1->data[p],scn2->data[p]);

  return (xor);
}


Scene  *Complement3(Scene *scn)
{
  Scene *cscn=NULL;
  int p,n,Imax;
  
  n = scn->xsize*scn->ysize*scn->zsize;
  cscn = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  Imax = MaximumValue3(scn);
  for (p=0; p < n; p++)
    cscn->data[p] = Imax - scn->data[p];
  return(cscn);
}

Scene *SQRT3(Scene *scn)
{
  Scene *sqrscn=NULL;
  int p,n;
  
  n = scn->xsize*scn->ysize*scn->zsize;
  sqrscn = CopyScene(scn);  
  for (p=0; p < n; p++){
    sqrscn->data[p] = (int)sqrt((double)sqrscn->data[p]);
  }
  return(sqrscn);
}

Scene *Add3(Scene *scn1, int value)
{
  Scene *sum;
  int p, n;

  sum = CreateScene(scn1->xsize, scn1->ysize, scn1->zsize);
  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for (p = 0; p < n; p++)
    sum->data[p] = scn1->data[p] + value;

  return (sum);
}

Scene *Mult3(Scene *scn1, Scene *scn2)
{
  Scene *mscn=CopyScene(scn1);
  int i,n=scn1->xsize*scn1->ysize*scn1->zsize;

  for (i=0; i < n; i++) 
    mscn->data[i]*=scn2->data[i];
  return(mscn);
}

Scene *Abs3(Scene *scn)
{
  Scene *absscn=NULL;
  int p,n;
  
  n = scn->xsize*scn->ysize*scn->zsize;
  absscn = CopyScene(scn);  
  for (p=0; p < n; p++){
    absscn->data[p] = abs(scn->data[p]);
  }
  return(absscn);
}



void Diff3inplace(Scene *scn1, Scene *scn2){
  int p,n;

  n = scn1->xsize*scn1->ysize*scn1->zsize;
  for(p=0; p<n; p++)
    scn1->data[p] = MAX(scn1->data[p] - scn2->data[p], 0);
}


void Sum3inplace(Scene *scn1, Scene *scn2){
  int p,n;

  n = scn1->xsize*scn1->ysize*scn1->zsize;
  for(p=0; p<n; p++)
    scn1->data[p] = scn1->data[p] + scn2->data[p];
}


void Or3inplace(Scene *scn1, Scene *scn2){
  int p, n;

  n = scn1->xsize * scn1->ysize * scn1->zsize;
  for(p=0; p<n; p++)
    scn1->data[p] = MAX(scn1->data[p],scn2->data[p]);
}


void Negate3inplace(Scene *scn){
  int p;

  for(p=0; p<scn->n; p++)
    scn->data[p] = -scn->data[p];
}



