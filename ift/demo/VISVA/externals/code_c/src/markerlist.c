
#include "markerlist.h"


MarkerList *CreateMarkerList(int nseeds){
  MarkerList *ML;

  ML = (MarkerList *) calloc(1,sizeof(MarkerList));
  if(ML == NULL)
    Error(MSG1,"CreateMarkerList");

  ML->data = (MarkerListNode *) calloc(nseeds, sizeof(MarkerListNode));
  if(ML->data == NULL)
    Error(MSG1,"CreateMarkerList");

  ML->n = nseeds;

  return ML;
}


void        DestroyMarkerList(MarkerList **ML){
  MarkerList *aux;
  
  aux = *ML;
  if(aux != NULL){
    free(aux->data);
    free(aux);
    *ML = NULL;
  }
}


//---------------------------------------
//---------------------------------------
//---------------------------------------


void  DrawMarkers3(Scene *scn, Set *S,
		   int value){
  Set *aux;
  int p;

  aux = S;
  while(aux != NULL){
    p = aux->elem;
    scn->data[p] = value;
    aux = aux->next;
  }
}


void LoadMarkers3(char *file, Scene *scn,
		  Set **Si, Set **Se){
  FILE *fp=NULL;
  int nseeds,i,x,y,z,id,lb,p;

  fp = fopen(file,"r");
  if(fp==NULL)
    Error(MSG2,"LoadMarkers3");

  fscanf(fp," %d \n",&nseeds);
  for(i=0;i<nseeds;i++){
    fscanf(fp," %d %d %d %d %d \n",&x,&y,&z,&id,&lb);
    if(ValidVoxel(scn,x,y,z)){
      p = VoxelAddress(scn,x,y,z);
      if(id>0){
	if(lb==0) InsertSet(Se, p);
	else      InsertSet(Si, p);
      }
    }
  }
  fclose(fp);
}


Set *Mask2Marker3(Scene *bin, AdjRel3 *A){
  Scene *border=NULL;
  Set *S=NULL;
  int p,n = bin->n;

  if(A==NULL){
    for(p=0; p<n; p++)
      if(bin->data[p]>0)
	InsertSet(&S, p);
  }
  else{
    border = GetSceneTransitions3(bin, A);
    S = Mask2Marker3(border, NULL);
    DestroyScene(&border);
  }
  return S;
}


float MaxRadiusByErosion3(Scene *bin){
  Scene *bin1=NULL,*edt=NULL;
  AdjRel3 *A=NULL;
  int Imax,Emax;

  A = Spheric(1.0);
  bin1 = Threshold3(bin, 0, 0);
  edt = Mask2EDT3(bin1, A, INTERIOR, INT_MAX, 0);
  Emax = MaximumValue3(edt);
  DestroyScene(&bin1);
  DestroyScene(&edt);

  bin1 = Threshold3(bin, 1, INT_MAX);
  edt = Mask2EDT3(bin1, A, INTERIOR, INT_MAX, 0);
  Imax = MaximumValue3(edt);
  DestroyScene(&bin1);
  DestroyScene(&edt);
  DestroyAdjRel3(&A);

  return MIN(sqrtf(Imax),sqrtf(Emax));
}


Scene *BkgMaskByErosion3(Scene *bin, float radius){
  Set *S=NULL;
  Scene *bin1=NULL,*bin2=NULL;

  bin1 = Threshold3(bin, 0, 0);
  bin2 = ErodeBin3(bin1, &S, radius);
  DestroyScene(&bin1);
  DestroySet(&S);
  return bin2;
}


Scene *ObjMaskByErosion3(Scene *bin, float radius){
  Set *S=NULL;
  Scene *bin1=NULL,*bin2=NULL;

  bin1 = Threshold3(bin, 1, INT_MAX);
  bin2 = ErodeBin3(bin1, &S, radius);
  DestroyScene(&bin1);
  DestroySet(&S);
  return bin2;
}



Set *BkgMarkerByErosion3(Scene *bin, float radius){
  Set *S=NULL,*Se=NULL;
  Scene *bin1=NULL,*bin2=NULL;

  bin1 = Threshold3(bin, 0, 0);
  bin2 = ErodeBin3(bin1, &S, radius);
  Se   = Mask2Marker3(bin2, NULL);
  DestroyScene(&bin1);
  DestroyScene(&bin2);
  DestroySet(&S);

  return Se;
}


Set *ObjMarkerByErosion3(Scene *bin, float radius){
  Set *S=NULL,*Si=NULL;
  Scene *bin1=NULL,*bin2=NULL;

  bin1 = Threshold3(bin, 1, INT_MAX);
  bin2 = ErodeBin3(bin1, &S, radius);
  Si   = Mask2Marker3(bin2, NULL);
  DestroyScene(&bin1);
  DestroyScene(&bin2);
  DestroySet(&S);

  return Si;
}

/*
Set  *ObjBandMarker3(Scene *bin,
		     float inner_radius,
		     float outer_radius){
  Set *S=NULL,*Se=NULL;
  Scene *bin1=NULL,*bin2=NULL,*bin3=NULL;

  bin1 = Threshold3(bin, 1, INT_MAX);
  bin2 = ErodeBin3(bin1, &S, inner_radius);
  DestroySet(&S);
  bin3 = ErodeBin3(bin1, &S, outer_radius);
  DestroySet(&S);
  DestroyScene(&bin1);
  bin1 = XOr3(bin2, bin3);
  Se   = Mask2Marker3(bin1, NULL);

  DestroyScene(&bin1);
  DestroyScene(&bin2);
  DestroyScene(&bin3);

  return Se;
}



Set  *BkgBandMarker3(Scene *bin,
		     float inner_radius,
		     float outer_radius){
  Set *S=NULL,*Se=NULL;
  Scene *bin1=NULL,*bin2=NULL,*bin3=NULL;

  bin1 = Threshold3(bin, 0, 0);
  bin2 = ErodeBin3(bin1, &S, inner_radius);
  DestroySet(&S);
  bin3 = ErodeBin3(bin1, &S, outer_radius);
  DestroySet(&S);
  DestroyScene(&bin1);
  bin1 = XOr3(bin2, bin3);
  Se   = Mask2Marker3(bin1, NULL);

  DestroyScene(&bin1);
  DestroyScene(&bin2);
  DestroyScene(&bin3);

  return Se;
}

*/
