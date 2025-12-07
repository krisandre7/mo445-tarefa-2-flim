
#include "evaluation3.h"


real DiceSimilarity3(Scene *mask1, 
		     Scene *mask2){
  real nelems_intersec,nelems_union;
  int  p,n;

  /* compute similarity between shapes */
  n = mask1->n;
  nelems_intersec = nelems_union = 0.0;
  for(p=0; p<n; p++){
    if(mask1->data[p]>0){
      nelems_union++;
      if(mask2->data[p]>0)
	nelems_intersec++;
    }
    else{
      if(mask2->data[p]>0)
	nelems_union++;
    }
  }

  return((2.0*nelems_intersec)/(nelems_union+nelems_intersec));
}


real    JaccardSimilarity3(Scene *mask1,
			   Scene *mask2){
  real nelems_intersec,nelems_union;
  int  p,n;

  /* compute similarity between shapes */
  n = mask1->n;
  nelems_intersec = nelems_union = 0.0;
  for(p=0; p<n; p++){
    if(mask1->data[p]>0){
      nelems_union++;
      if(mask2->data[p]>0)
	nelems_intersec++;
    }
    else{
      if(mask2->data[p]>0)
	nelems_union++;
    }
  }

  return(nelems_intersec/nelems_union);
}


real    CentrJaccardSimilarity3(Scene *mask1,
				Scene *mask2){
  real nelems_intersec,nelems_union;
  int  p,q,n;
  Voxel C1,C2,u,v;

  ComputeMaskCentroid3(mask1, &C1);
  ComputeMaskCentroid3(mask2, &C2);

  /* compute similarity between shapes */
  n = mask1->n;
  nelems_intersec = nelems_union = 0.0;

  for(p=0; p<n; p++){
    u.x = VoxelX(mask1, p);
    u.y = VoxelY(mask1, p);
    u.z = VoxelZ(mask1, p);

    v.x = u.x + (C2.x - C1.x);
    v.y = u.y + (C2.y - C1.y);
    v.z = u.z + (C2.z - C1.z);

    if(mask1->data[p]>0){
      nelems_union++;
      if(ValidVoxel(mask2, v.x, v.y, v.z)){
	q = VoxelAddress(mask2,v.x,v.y,v.z);
	if(mask2->data[q]>0)
	  nelems_intersec++;
      }
    }
    else{
      if(ValidVoxel(mask2, v.x, v.y, v.z)){
	q = VoxelAddress(mask2,v.x,v.y,v.z);
	if(mask2->data[q]>0)
	  nelems_union++;
      }
    }
  }

  return(nelems_intersec/nelems_union);
}



real    CentrDiceSimilarity3(Scene *mask1,
			     Scene *mask2){
  real nelems_intersec,nelems_union;
  int  p,q,n;
  Voxel C1,C2,u,v;

  ComputeMaskCentroid3(mask1, &C1);
  ComputeMaskCentroid3(mask2, &C2);

  /* compute similarity between shapes */
  n = mask1->n;
  nelems_intersec = nelems_union = 0.0;

  for(p=0; p<n; p++){
    u.x = VoxelX(mask1, p);
    u.y = VoxelY(mask1, p);
    u.z = VoxelZ(mask1, p);

    v.x = u.x + (C2.x - C1.x);
    v.y = u.y + (C2.y - C1.y);
    v.z = u.z + (C2.z - C1.z);

    if(mask1->data[p]>0){
      nelems_union++;
      if(ValidVoxel(mask2, v.x, v.y, v.z)){
	q = VoxelAddress(mask2,v.x,v.y,v.z);
	if(mask2->data[q]>0)
	  nelems_intersec++;
      }
    }
    else{
      if(ValidVoxel(mask2, v.x, v.y, v.z)){
	q = VoxelAddress(mask2,v.x,v.y,v.z);
	if(mask2->data[q]>0)
	  nelems_union++;
      }
    }
  }

  return((2.0*nelems_intersec)/(nelems_union+nelems_intersec));
}



real    CentrMObjDiceSimilarity3(Scene *label1,
				 Scene *label2){
  Scene *bin1=NULL,*bin2=NULL;
  real nelems_intersec,nelems_union;
  real dice,sum=0.0;
  int p,q,l,Lmax,Ln=0;
  int *hist1=NULL,*hist2=NULL;
  Voxel C1,C2,u,v;

  //----------------
  bin1 = Threshold3(label1, 1, INT_MAX);
  bin2 = Threshold3(label2, 1, INT_MAX);
  ComputeMaskCentroid3(bin1, &C1);
  ComputeMaskCentroid3(bin2, &C2);
  DestroyScene(&bin1);
  DestroyScene(&bin2);
  //----------------

  Lmax = MAX(MaximumValue3(label1),
	     MaximumValue3(label2));
  hist1 = AllocIntArray(Lmax+1);
  hist2 = AllocIntArray(Lmax+1);

  for(p=0; p<label1->n; p++)
    hist1[label1->data[p]]++;
  for(p=0; p<label2->n; p++)
    hist2[label2->data[p]]++;
  //----------------

  for(l=1; l<=Lmax; l++){
    if(hist1[l]==0 && hist2[l]==0)
      continue;

    bin1 = Threshold3(label1, l, l);
    bin2 = Threshold3(label2, l, l);

    //----------------
    //Dice Similarity3(bin1, bin2):

    nelems_intersec = nelems_union = 0.0;
    for(p=0; p<bin1->n; p++){
      u.x = VoxelX(bin1, p);
      u.y = VoxelY(bin1, p);
      u.z = VoxelZ(bin1, p);

      v.x = u.x + (C2.x - C1.x);
      v.y = u.y + (C2.y - C1.y);
      v.z = u.z + (C2.z - C1.z);

      if(bin1->data[p]>0){
	nelems_union++;
	if(ValidVoxel(bin2, v.x, v.y, v.z)){
	  q = VoxelAddress(bin2,v.x,v.y,v.z);
	  if(bin2->data[q]>0)
	    nelems_intersec++;
	}
      }
      else{
	if(ValidVoxel(bin2, v.x, v.y, v.z)){
	  q = VoxelAddress(bin2,v.x,v.y,v.z);
	  if(bin2->data[q]>0)
	    nelems_union++;
	}
      }
    }

    dice = (2.0*nelems_intersec)/(nelems_union+nelems_intersec);
    //----------------    
    Ln++;
    sum += dice;
    DestroyScene(&bin1);
    DestroyScene(&bin2);
  }

  //----------------
  free(hist1);
  free(hist2);

  return (sum/(real)Ln);
}



//mask1: Ground Truth
//mask2: Segmentation Result
int     AssessTP3(Scene *mask1, Scene *mask2){
  int p,n,tp=0;

  n = mask1->n; 
  for(p=0; p<n; p++){
    if(mask1->data[p]>0 && mask2->data[p]>0)
      tp++;
  }
  return tp;
}


//mask1: Ground Truth
//mask2: Segmentation Result
int     AssessFN3(Scene *mask1, Scene *mask2){
  int p,n,fn=0;

  n = mask1->n;
  for(p=0; p<n; p++){
    if(mask1->data[p]>0 && mask2->data[p]==0)
      fn++;
  }
  return fn;
}

//mask1: Ground Truth
//mask2: Segmentation Result
int     AssessFP3(Scene *mask1, Scene *mask2){
  int p,n,fp=0;

  n = mask1->n;
  for(p=0; p<n; p++){
    if(mask1->data[p]==0 && mask2->data[p]>0)
      fp++;
  }
  return fp;
}


//mask1: Ground Truth
//mask2: Segmentation Result
int     AssessTN3(Scene *mask1, Scene *mask2){
  int p,n,tn=0;

  n = mask1->n;
  for(p=0; p<n; p++){
    if(mask1->data[p]==0 && mask2->data[p]==0)
      tn++;
  }
  return tn;
}



