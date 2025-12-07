#include "ift.h"

int main(int argc, char* argv []) {

  if(argc!=2) {
    iftError("Usage: iftCharacterSeparation <plate.pgm> ", argv[0]);
  }

  iftImage   *aux   = iftReadImageP5(argv[1]);
  iftImage   *plate = iftEqualize(aux,255);
  iftDestroyImage(&aux);
   int mean=0;
    for (int p=0; p < plate->n; p++) {
      mean += plate->val[p];
    }
    mean /= plate->n;

  iftImage   *vclose  = iftVolumeClose(plate,5000);
  iftAdjRel  *A       = iftCircular(1.5);
  iftImage   *label   = iftWaterGray(plate,vclose,A);
  for (int i=1; i <= iftMaximumValue(label); i++) {
    iftBoundingBox mbb = iftMinObjectBoundingBox(label, i, NULL);
    iftVoxel  pos = mbb.begin;
    iftImage *comp = iftExtractROI(label, mbb);
    iftSetImage(comp,mean);
    for (int p=0; p < plate->n; p++) {
      if (label->val[p]==i) {
	iftVoxel v,u = iftGetVoxelCoord(plate,p);
	v.x = u.x - pos.x; 
	v.y = u.y - pos.y;
	v.z = 0; 
	int q = iftGetVoxelIndex(comp,v);
	comp->val[q]=plate->val[p];
      }      
    }
    char filename[200];
    sprintf(filename,"character%d.pgm",i);
    iftWriteImageP5(comp,filename);
    iftDestroyImage(&comp);
  }
 
  iftDestroyImage(&plate);
  iftDestroyImage(&vclose);
  iftDestroyImage(&label);
  iftDestroyAdjRel(&A);
  
  return(0);
}
