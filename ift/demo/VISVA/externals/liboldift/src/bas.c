#include "bas.h"

/**********************************************/
/* BAS - Beam Angle Statistics )*/
/**********************************************/


/***************BAS EXTRACTION ALGORITHM*********************/

/*resample curve*/
representation_type *resample(representation_type *curve, int nsamples)
{
  representation_type *rcurve;
  Image *img1, *img2;
  int x;
  
  img1= CreateImage(curve->length, 3);
  
  for (x=0; x<curve->length; x++){
    img1->val[x] = curve->mean[x];
    img1->val[x+img1->tbrow[1]] = curve->second[x];
    img1->val[x+img1->tbrow[2]] = curve->third[x];
  }
  
  img2 = Scale(img1, (((float) nsamples)/curve->length), 1);
  
  rcurve = (representation_type *) calloc(1, sizeof(representation_type));
  rcurve->length = nsamples;
  rcurve->mean = (int *) calloc(nsamples, sizeof(int));
  rcurve->second =(int *) calloc(nsamples, sizeof(int));
  rcurve->third = (int *) calloc(nsamples, sizeof(int));
  for (x=0; x<nsamples; x++){
    rcurve->mean[x] = img2->val[x];
    rcurve->second[x] = img2->val[x+img2->tbrow[1]];
    rcurve->third[x] = img2->val[x+img2->tbrow[2]];
  }
  
  DestroyImage(&img1);
  DestroyImage(&img2);
  return(rcurve);
}

/*finds angle between two beams*/
double find_angle(deltax,deltay)
     int deltax;
     int deltay;
{
  double angle;
  double pi;
  
  pi=22.0/7.0; 
  
  if((deltax==0) && (deltay==0))
    angle=0.0;
  else{
    angle=atan((10.0*abs(deltax))/(10.0*abs(deltay)));
    angle=angle*180.0/pi;  
    if((deltax <= 0) && (deltay >= 0)) 
      angle=360.0-angle;
    else if((deltax <= 0) && (deltay <=0)) 
      angle=180.0 + angle;  
    else if((deltax >= 0) && (deltay <=0)) 
      angle=180.0 - angle;
  }
  
  return(angle);
}

/***    input  : boundary of shape in boundary_type                           ***/
/***    output : BAS function in representation type (mean, second and third) ***/
representation_type *extract_feature(boun)
     boundary_type *boun;
{
  representation_type *curve_feature; 
  int i,j,x1,x2,x3,y1,y2,y3,curvelength;
  double angle_1,angle_2,curve,total,previous;
  int delta_x, delta_y,mean,second,third;
  int *bearing_array;
  
  curve_feature = (representation_type *) calloc(1, sizeof(representation_type));
  curve_feature->length=boun->length;
  curve_feature->mean= (int *) calloc(boun->length, sizeof(int));
  curve_feature->second=(int *) calloc(boun->length, sizeof(int));
  curve_feature->third=(int *) calloc(boun->length, sizeof(int));
  
  
  curvelength=(int)(boun->length/2);
  bearing_array=(int *) calloc((curvelength-1), sizeof(int));
  for(i=0; i<boun->length; i++){
    total=0.0;
    x1=boun->X[((i-1)+boun->length)%boun->length];
    y1=boun->Y[((i-1)+boun->length)%boun->length];
    x2=boun->X[i];
    y2=boun->Y[i];
    x3=boun->X[((i+1)+boun->length)%boun->length];
    y3=boun->Y[((i+1)+boun->length)%boun->length];
  
    delta_x=x1-x2;
    delta_y=-(y1-y2);
    angle_1=find_angle(delta_x,delta_y);
    delta_x=x3-x2;
    delta_y=-(y3-y2);
    angle_2=find_angle(delta_x,delta_y);
    if(angle_1 >= angle_2)
      curve=angle_1-angle_2;
    else
      curve=360.0 + angle_1-angle_2;
  
    total+=curve;
    bearing_array[0]=(int)curve;
    previous=curve;
    for(j=2; j<curvelength; j++){
      x1=boun->X[((i-j)+boun->length)%boun->length];
      y1=boun->Y[((i-j)+boun->length)%boun->length];
      x2=boun->X[i];
      y2=boun->Y[i];
      x3=boun->X[((i+j)+boun->length)%boun->length];
      y3=boun->Y[((i+j)+boun->length)%boun->length];
      delta_x=x1-x2;
      delta_y=-(y1-y2);
      angle_1=find_angle(delta_x,delta_y);
      delta_x=x3-x2;
      delta_y=-(y3-y2);
      angle_2=find_angle(delta_x,delta_y);
      if(angle_1 >= angle_2)
	curve=angle_1-angle_2;
      else
	curve=360.0 + angle_1-angle_2;
      
      if(j > 3){
	if(((curve-previous) > 180))
	  curve=curve-360.0;
	else
	  if(((previous-curve) > 180))
	    curve=curve+360.0;
      }
      
      bearing_array[j-1]=(int)curve; 
      total+=curve;
      previous=curve;
    }
    
    mean=(int)(total/(double)(curvelength-1));
    total=0.0;
    for(j=0;j<curvelength-1; j++)
      total+=pow((bearing_array[j]-mean),2.0);
    second=pow(total/(double)(curvelength-2),0.5);
    total=0.0;
    for(j=0;j<curvelength-1; j++)
      total+=pow(abs(bearing_array[j]-mean),3.0);
    third=pow(total/(double)(curvelength-2),(1.0/3.0)); 
  
    curve_feature->mean[i]=mean;
    curve_feature->second[i]=second;
    curve_feature->third[i]=third;
  }    
  free(bearing_array);
  return(curve_feature);
}

Curve *BAS(Image *in,int rsp,int nsamples){
  Curve *featurevector = NULL;
  Curve *contour = NULL;
  Curve *moment1 = NULL;
  Curve *moment2 = NULL;
  Curve *moment3 = NULL;
  
  boundary_type *bound;
  representation_type *curve, *rcurve;
  int i;
  
  //timer  tic,toc;
  
  //gettimeofday(&tic,NULL);
  
  //printf("Reading Contour...\n");
  contour = Image2Curve(in);
    
  bound = (boundary_type *) calloc(1, sizeof(boundary_type));
  bound->length = contour->n;
  bound->X = (int *)calloc(bound->length, sizeof(int));
  bound->Y = (int *)calloc(bound->length, sizeof(int));
  for (i=0; i<bound->length; i++){
    bound->X[i] = (int)contour->X[i];
    bound->Y[i] = (int)contour->Y[i];
  }
  //printf("Done.\n");
  
  //printf("Computing Moments...\n");
  curve = extract_feature(bound);

  if(rsp == 0)
  {
  	rcurve = resample(curve, curve->length);
	nsamples = curve->length;
  }
  else
	rcurve = resample(curve, nsamples);

  moment1 = CreateCurve(nsamples);
  moment2 = CreateCurve(nsamples);
  moment3 = CreateCurve(nsamples);
  
  featurevector = CreateCurve(3*nsamples);
  for (i=0; i<3*nsamples; i++){
    featurevector->X[i] = (double)i;
  }
  
  for (i=0; i<nsamples; i++){
    moment1->X[i]= moment2->X[i]= moment3->X[i]=(double)i;
    featurevector->Y[i] = moment1->Y[i]= (double) rcurve->mean[i];
    featurevector->Y[nsamples+i] = moment2->Y[i]= (double) rcurve->second[i];
    featurevector->Y[2*nsamples+i] = moment3->Y[i]= (double) rcurve->third[i];
  }

/*  WriteCurve(moment1,"moment1.out");
  WriteCurve(moment2,"moment2.out");
  WriteCurve(moment3,"moment3.out");*/
  //printf("Done.\n");
  
 // printf("Deallocating Structures...\n");
  free(bound->X);
  free(bound->Y);
  free(bound);
  free(curve->mean);
  free(curve->second);
  free(curve->third);
  free(curve);
  free(rcurve->mean);
  free(rcurve->second);
  free(rcurve->third);
  free(rcurve);
  DestroyCurve(&contour);
  DestroyCurve(&moment1);
  DestroyCurve(&moment2);
  DestroyCurve(&moment3);
  printf("Done.\n");
  
  //gettimeofday(&toc,NULL);
  //printf("Time in msecs %f\n",(toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001);

  return featurevector;
}

FeatureVector1D *BAS_ExtractionAlgorithm(Image *in,int rsp,int nsamples){
 Curve *curve = NULL;
 FeatureVector1D *fv = NULL;
 
 curve = BAS(in,rsp,nsamples);
 fv = CurveTo1DFeatureVector(curve);
 
 DestroyCurve(&curve);
 return fv;
}
/************************************************************/



/***************BAS SIMILARITY ALGORITHM*********************/

long mymin(Dist1,Dist2,Dist3)
     long Dist1;
     long Dist2;
     long Dist3;
{
  if((Dist1<=Dist2) && (Dist1<=Dist3)) 
    return(Dist1);
  else if((Dist2<=Dist1) && (Dist2<=Dist3)) 
    return(Dist2);
  /*else if((Dist3<=Dist1) && (Dist3<=Dist2)) */
  return(Dist3);
}


long Cum_Dist_Optimal(fv1,fv2,dim1,dim2,DISTANCE)
     representation_type fv1;
     representation_type fv2;
     int dim1;
     int dim2;
     long *DISTANCE;
{
  long temp_dist;
  int i,j;
  int penalty;
  
  penalty=300;
  /* OPTIMAL CORRESPONDENCE OF STRINGS
   */
  DISTANCE[0*(dim2+1)+0]=0;
  for(j=1;j<=dim2;j++)
    DISTANCE[0*(dim2+1)+j]=j * penalty;
  
  for(i=1;i<=dim1;i++)
    DISTANCE[i*(dim2+1)+0]=i * penalty;
  
  for(i=1;i<=dim1;i++)
    for(j=1;j<=dim2;j++)
      if(abs(i-j) < (5)){
	temp_dist=abs(fv1.mean[i-1]-fv2.mean[j-1]) +
	  abs(fv1.second[i-1]-fv2.second[j-1]) +
	  abs(fv1.third[i-1]-fv2.third[j-1]);
	
	DISTANCE[i*(dim2+1)+j]= temp_dist +
	  mymin(DISTANCE[(i-1)*(dim2+1)+(j-1)],
	      DISTANCE[(i-1)*(dim2+1)+(j)] + penalty,
	      DISTANCE[(i)*(dim2+1)+(j-1)] + penalty); 
      }
  return(DISTANCE[(dim1)*(dim2+1)+(dim2)]/dim2);
}

long find_distance(representation_type fv1, representation_type fv2,
		   int dim1, int dim2)
{
  long distance,k,i,j,temp_dist;
  representation_type temp_list1, temp_list2;
  long *DISTANCE;
  
  DISTANCE=(long *) calloc((dim1+1)*(dim2+1),sizeof(long));
  
  temp_list1.mean=(int *) calloc(dim2,sizeof(int));
  temp_list1.second=(int *) calloc(dim2,sizeof(int));
  temp_list1.third=(int *) calloc(dim2,sizeof(int));
  temp_list1.length=0;
  temp_list2.mean=(int *) calloc(dim2,sizeof(int));
  temp_list2.second=(int *) calloc(dim2,sizeof(int));
  temp_list2.third=(int *) calloc(dim2,sizeof(int));
  
  temp_dist=10000000;
  
  for(i=0;i<dim1+1;i++)
    for(j=0;j<dim2+1;j++)
      DISTANCE[i*(dim2+1)+j]=10000000;
  
  for(k=0; k<dim2; k++){
    for(i=0;i<dim2;i++){
      temp_list1.mean[i]=fv2.mean[(i+k)%dim2];
      temp_list1.second[i]=fv2.second[(i+k)%dim2];
      temp_list1.third[i]=fv2.third[(i+k)%dim2]; 
    }   
    distance=Cum_Dist_Optimal(fv1,temp_list1,dim1,dim2,DISTANCE);
    if(temp_dist>distance) temp_dist=distance;
  }
  /***Taking the mirror of fv2 *****/
  
  for(i=0;i<dim2;i++){
    temp_list2.mean[i]=fv2.mean[(dim2-1)-i];
    temp_list2.second[i]=fv2.second[(dim2-1)-i];
    temp_list2.third[i]=fv2.third[(dim2-1)-i]; 
  }
  
  for(k=0; k<dim2; k++){
    for(i=0;i<dim2;i++){
      temp_list1.mean[i]=temp_list2.mean[(i+k)%dim2];
      temp_list1.second[i]=temp_list2.second[(i+k)%dim2];
      temp_list1.third[i]=temp_list2.third[(i+k)%dim2]; 
    }
    distance=Cum_Dist_Optimal(fv1,temp_list1,dim1,dim2,DISTANCE);
    if(temp_dist>distance) temp_dist=distance;
  }
  
  distance=temp_dist;
  
  free(temp_list1.mean);
  free(temp_list1.second);
  free(temp_list1.third);
  free(temp_list2.mean);
  free(temp_list2.second);
  free(temp_list2.third);
  free(DISTANCE);
  return(distance);
}

// To compute distance between two BAS feature vectors - USING OCS
double BAS_SimilarityAlgorithm(FeatureVector1D *c1, FeatureVector1D *c2){
  
  representation_type fv1, fv2;
  int n, m, i;
  long dist;
  
  n = c1->n/3;
  m = c2->n/3;
  fv1.mean=(int *) calloc(n,sizeof(int));
  fv1.second=(int *) calloc(n,sizeof(int));
  fv1.third=(int *) calloc(n,sizeof(int));
  fv1.length=0;
  fv2.mean=(int *) calloc(m,sizeof(int));
  fv2.second=(int *) calloc(m,sizeof(int));
  fv2.third=(int *) calloc(m,sizeof(int));  
  fv2.length=0;
  
  for (i=0; i<n; i++){
    fv1.mean[i] = (int)c1->X[i];
    fv1.second[i] = (int)c1->X[n + i];
    fv1.third[i] = (int)c1->X[2*n + i];
  }

  for (i=0; i<m; i++){
    fv2.mean[i] = (int)c2->X[i];
    fv2.second[i] = (int)c2->X[m + i];
    fv2.third[i] = (int)c2->X[2*m + i];
  }
  
  dist = find_distance(fv1,fv2, n, m);
  
  free(fv1.mean);
  free(fv1.second);
  free(fv1.third);
  free(fv2.mean);
  free(fv2.second);
  free(fv2.third);
  
  return(dist);
}
/************************************************************/
