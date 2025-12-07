#include "ift.h"

//Difference between object fractions in both sides
float iftSymmetric(iftImage *bin)
{
   int halfX, halfY, halfZ;
   float AIF = 0, ADF = 0, aIF = 0, aDF = 0, AIE = 0, ADE = 0, aIE = 0, aDE = 0, max, min, total = 0, res;
   iftVoxel u, c;
  
   c = iftGeometricCenterVoxel(bin);
   halfX = c.x-1;
   halfY = c.y-1;
   halfZ = c.z-1;

   for (int i=0; i < bin->n; i++) 
   {
      u = iftGetVoxelCoord(bin, i);
      if (bin->val[i]==1) 
      {
         if ((u.x <= halfX) && (u.y <= halfY) && (u.z <= halfZ))
            AIF++;
         if ((u.x > halfX) && (u.y <= halfY) && (u.z <= halfZ))
            ADF++;
         if ((u.x <= halfX) && (u.y > halfY) && (u.z <= halfZ))
            aIF++;
         if ((u.x > halfX) && (u.y > halfY) && (u.z <= halfZ))
            aDF++;
         if ((u.x <= halfX) && (u.y <= halfY) && (u.z > halfZ))
            AIE++;
         if ((u.x > halfX) && (u.y <= halfY) && (u.z > halfZ))
            ADE++;
         if ((u.x <= halfX) && (u.y > halfY) && (u.z > halfZ))
            aIE++;
         if ((u.x > halfX) && (u.y > halfY) && (u.z > halfZ))
            aDE++;
         total++;
      }
   }

   min = MIN(AIF, MIN(ADF, MIN (aIF, MIN (aDF,MIN(AIE, MIN(ADE, MIN(aIE, aDE)))))));
   max = MAX(AIF, MAX(ADF, MAX (aIF, MAX (aDF,MAX(AIE, MAX(ADE, MAX(aIE, aDE)))))));
   printf("%5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f\n", AIF, ADF, aIF, aDF, AIE, ADE, aIE, aDE);
   res = abs(max - min) * 100.0 / total;
   
   return res;   
}

float iftSphericity(iftImage *bin)
{
  //Better try 2D circles, because image is not spheric
  iftSet *baseBorders;
  iftAdjRel *AMax, *A = iftCircular(1.0);
  int voxelId, maxDist, q, count, dist, totalCount, cantSlices = 0;
  float res, resTotal = 0.0;
  iftVoxel u, center,v;
  iftImage *slice;

  center.x = bin->xsize/2; center.y = bin->ysize/2; center.z = 0;

  for (int z=0; z < bin->zsize; z++) 
  {
     count = 0;
     totalCount = 0;
     res = 0;
     slice = iftGetXYSlice(bin, z);
     baseBorders = iftObjectBorderSet(slice, A);
     maxDist = INFINITY_INT_NEG;

     while(baseBorders)
     {        
        voxelId = baseBorders->elem;
        u = iftGetVoxelCoord(slice, voxelId);
        dist = (u.x-center.x)*(u.x-center.x) + (u.y-center.y)*(u.y-center.y);
        if (dist > maxDist)
           maxDist = dist;
        baseBorders = baseBorders->next;
     }

     AMax = iftCircular(sqrt(maxDist));

     for(int i = 0; i < AMax->n; i++)
     {
        v = iftGetAdjacentVoxel(AMax, center, i);
        if (iftValidVoxel(slice, v))
        { 
          q = iftGetVoxelIndex(slice, v);
          if (slice->val[q] > 0)
             count++;   
          totalCount++;            
        }
     }

     if (totalCount == 0) 
        res = 0.0;
     else
     {
        res = count * 1.0 / totalCount;
        cantSlices++;
     }

     resTotal += res;

     iftDestroyImage(&slice);
     iftDestroyAdjRel(&AMax);
     iftDestroySet(&baseBorders);
   }  

   resTotal /= (cantSlices * 1.0);
   iftDestroyAdjRel(&A);
   return resTotal;        
}

int main(int argc, char *argv[]) 
{
  iftImage  *orig, *aux, *grad, *res, *extract;
  timer     *t1=NULL,*t2=NULL;
  iftAdjRel *A26, *A8;
  iftLabeledSet *S=NULL;
  int i, tolerance = 20;
  iftVoxel v, uo, uf; 
  float sphereMeasure, symmetricMeasure;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*-----------------------------40---------------------------*/

  if (argc!=6)
    iftError("Usage: iftClassifyNodules <thorax.scn> <x> <y> <z> <res.scn>","main");
  
  t1       = iftTic();

  printf("%d %d %d\n", atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  orig = iftReadImage(argv[1]);
  A26 = iftSpheric(sqrtf(3.0));
  A8 = iftSpheric(sqrtf(1.0));
  v.x = atoi(argv[2]); v.y = atoi(argv[3]); v.z = atoi(argv[4]);
  uo.x = MAX(v.x-tolerance, 0);
  uo.y = MAX(v.y-tolerance, 0);
  uo.z = MAX(v.z-tolerance, 0);
  uf.x = MIN(v.x+tolerance, orig->xsize-1);
  uf.y = MIN(v.y+tolerance, orig->ysize-1);
  uf.z = MIN(v.z+tolerance, orig->zsize-1);
  extract = iftExtractROI(orig, uo, uf);
  aux = iftThreshold(extract,0,iftOtsu(extract)/2,1);
  grad = iftImageGradientMagnitude(extract, A8);
  v.x-=uo.x; v.y-=uo.y; v.z-=uo.z;
  i = iftGetVoxelIndex(extract,v);
  iftInsertLabeledSet(&S, i, 1);
  iftInsertLabeledSet(&S, i, 1);
  iftWriteImage(grad,"grad.scn");
  iftWriteImage(aux,"thres.scn");
  for(i = 0; i < aux->n; i++)
     if (aux->val[i] == 1)
        iftInsertLabeledSet(&S, i, 0);
  res = iftWatershed(grad,A26,S, NULL);
  iftWriteImage(res,argv[5]);
 
  sphereMeasure = iftSphericity(res); 
  symmetricMeasure = iftSymmetric(res);

  printf("Volume: %5.2f Spherical: %5.2f Symmetric: %5.2f\n", iftObjectVolume(res,1), sphereMeasure, symmetricMeasure);

  iftDestroyImage(&aux);
  iftDestroyImage(&orig);
  iftDestroyImage(&grad);
  iftDestroyImage(&res);
  iftDestroyImage(&extract);
  iftDestroyAdjRel(&A26);
  iftDestroyAdjRel(&A8);
  iftDestroyLabeledSet(&S);

  t2     = iftToc();
  fprintf(stdout,"Thorax segmented in %f ms\n",iftCompTime(t1,t2));



  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  return(0);

}
