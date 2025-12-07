#include "ift.h"

iftImage *iftBrainMask(iftImage *label)
{
  for (int p=0; p < label->n; p++)
    if (label->val[p]==2)
      label->val[p]=0;
    else
      label->val[p]=1;

  iftImage *bin1 = iftAddFrame(label,15,0);
  iftImage *bin2 = iftCloseBin(bin1,15.0);
  iftDestroyImage(&bin1);
  bin1  = iftRemFrame(bin2,15);
  iftDestroyImage(&bin2);
  iftAdjRel *B   = iftSpheric(sqrtf(3.0));
  iftImage *tde = iftEuclDistTrans(bin1,B,IFT_INTERIOR, NULL, NULL, NULL);
  iftDestroyAdjRel(&B);
  int Dmax = iftMaximumValue(tde);
  iftVoxel center;
  center.x = center.y = center.z = 0; 
  for (int p=0; p < label->n; p++)
    if (tde->val[p]==Dmax){
      center = iftGetVoxelCoord(tde,p);
      break;
    }
  iftDestroyImage(&tde);
  for (int p=0; p < label->n; p++){
    iftVoxel u = iftGetVoxelCoord(label,p);
    if (iftSquaredVoxelDistance(u,center) > 1.5*Dmax)
      bin1->val[p]=0;
  }
  return(bin1);
}

iftImage *iftCorrectInhomogeneity(iftImage *img, iftImage *mask, float alpha, float adj_radius) {
    int Imax = iftMaximumValue(img);

    /* Compute inhomogeneity correction inside the envelop */

    iftImage *corr = iftCopyImage(img);
    iftAdjRel *A   = iftSpheric(adj_radius);

    if (mask != NULL) {
        puts("************************");
        #pragma omp parallel for schedule(auto)
        for (int p =0; p < img->n; p++) {
            if (mask->val[p]==1) {      
                iftVoxel u = iftGetVoxelCoord(img,p);
                int Ir     = img->val[p]; 

                for (int i =1; i < A->n; i++) {
                    iftVoxel v = iftGetAdjacentVoxel(A,u,i);
                    if (iftValidVoxel(img,v)) {
                    int q = iftGetVoxelIndex(img,v);

                    if (img->val[q]>Ir)
                      Ir = img->val[q];
                    }
                }
                if (Ir > 0) {
                    float f = pow(((float)Imax/(float)Ir),alpha);
                    corr->val[p] = img->val[p]*f;
                } 
            } else corr->val[p]=0;
        }
    } else {
        #pragma omp parallel for schedule(auto)
        for (int p =0; p < img->n; p++) {
            iftVoxel u = iftGetVoxelCoord(img,p);
            int Ir     = img->val[p];

            for (int i =1; i < A->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(A,u,i);

                if (iftValidVoxel(img,v)) {
                    int q = iftGetVoxelIndex(img,v);
                    
                    if (img->val[q]>Ir)
                        Ir = img->val[q];
                }
            }
            if (Ir > 300) {
                float f = pow(((float)Imax/(float)Ir),alpha);
                corr->val[p] = img->val[p]*f;
            }       
        }
    }

    iftDestroyAdjRel(&A);

    iftImage *norm = iftNormalizeWithNoOutliers(corr, 0, 4095, 0.98);
    iftDestroyImage(&corr);
    corr = norm;

    return(corr);
}

int main(int argc, char *argv[]) 
{
  iftImage    *img=NULL,*label=NULL,*corr=NULL;
  timer       *t1=NULL,*t2=NULL;


  if ((argc!=5)&&(argc!=6))
    iftError("Usage: iftCorrectInhomogeneity <input-original-image.scn> <input-csf-label.scn (optional)> <alpha (e.g., > 1.0)> <adjacency radius (e.g., 3.0)> <output-corrected-image.scn>","main");

  img   = iftReadImageByExt(argv[1]);
  if (argc==6)
    label = iftReadImageByExt(argv[2]);
  else
    label = NULL;
  
  t1   = iftTic();
  if (label != NULL){
    iftSet *S = NULL;
    iftImage *mask = iftDilateBin(label, &S, 9.0);
    // iftImage *mask = iftBrainMask(label);
    // corr = iftCorrectInhomogeneity(img,mask,atof(argv[3]),atof(argv[4]));
    // iftDestroyImage(&mask);
    corr = iftCorrectInhomogeneity(img,mask,atof(argv[3]),atof(argv[4]));
  } else
    corr = iftCorrectInhomogeneity(img,NULL,atof(argv[2]),atof(argv[3]));
  t2   = iftTic();
  printf("Inhomogeneity Corrected in: %.2f ms\n", iftCompTime(t1, t2));

  if (argc == 6){
    iftWriteImageByExt(corr,argv[5]);
    iftDestroyImage(&label);
  }else
    iftWriteImageByExt(corr,argv[4]);
    
  iftDestroyImage(&img);
  iftDestroyImage(&corr);



  return(0);

}
