#include "ift.h"

int main(int argc, const char *argv[])
{
    iftImage  *img,*label,*rend[3];
    iftFImage *scene;
    iftGraphicalContext *gc;
    timer     *t1=NULL,*t2=NULL;
    float      tilt, spin;

    /*--------------------------------------------------------*/
    size_t MemDinInicial, MemDinFinal;
    MemDinInicial = iftMemoryUsed();

    /*--------------------------------------------------------*/

    if (argc != 6) {
      printf("iftStereo P1 P2 P3 P4 P5\n");
      printf("P1 - input original scene (.scn)\n");
      printf("P2 - input label scene (.scn)\n");
      printf("P3 - input tilt angle (e.g., 45)\n");
      printf("P4 - input spin angle (e.g., 45) \n");
      printf("P5 - output rendition (.png)\n");
      exit(0);
    }
    
    img    = iftReadImageByExt(argv[1]);
    label  = iftReadImageByExt(argv[2]);
    scene  = iftImageToFImage(img);
    iftDestroyImage(&img);
    tilt = atof(argv[3]);
    spin = atof(argv[4]);

    t1  = iftTic();
    
    gc    = iftCreateGraphicalContext(scene,label);
    iftSetViewDir(gc,tilt,spin);
    iftSetObjectNormal(gc, SCENE_NORMAL);
    //iftSetObjectNormal(gc, OBJECT_NORMAL);
    int nobjs = iftMaximumValue(label);
    for (int l=1; l <= nobjs; l++){
      iftSetObjectColor(gc, l, 1.0, 1.0, 1.0);
      iftSetObjectVisibility(gc, l, 1);
      iftSetObjectOpacity(gc, l, 1.0);
    }
    iftSetProjectionMode(gc,RAYCASTING);

    t2     = iftToc();

    fprintf(stdout,"Preprocessing in %f ms\n",iftCompTime(t1,t2));

    t1 = iftTic();


    iftSetViewDir(gc,tilt,spin-5);
    img     = iftSurfaceRender(gc);
    rend[0] = iftNormalize(img, 0, 255);
    iftDestroyImage(&img);
    
    iftSetViewDir(gc,tilt,spin+5);
    img = iftSurfaceRender(gc);
    rend[2] = iftNormalize(img, 0, 255);
    iftDestroyImage(&img);

    rend[1] = iftCreateImageFromImage(rend[0]);
    for (int p = 0; p < rend[1]->n; p++) {
      rend[1]->val[p] = (rend[0]->val[p]+rend[2]->val[p])/2;
    }
        
    img = iftCreateColorImage(rend[0]->xsize,rend[0]->ysize,rend[0]->zsize,8);
    for (int p = 0; p < img->n; p++) {
      iftColor rgb, ycbcr;
      rgb.val[0] = rend[0]->val[p];
      rgb.val[1] = rend[1]->val[p];
      rgb.val[2] = rend[2]->val[p];
      ycbcr      = iftRGBtoYCbCr(rgb,255);
      img->val[p] = ycbcr.val[0];
      img->Cb[p]  = ycbcr.val[1];
      img->Cr[p]  = ycbcr.val[2];
    }
    
    t2 = iftToc();

    fprintf(stdout, "Rendering in %f ms\n", iftCompTime(t1, t2));

    iftWriteImageByExt(img, argv[5]);

    iftDestroyImage(&label);
    iftDestroyFImage(&scene);
    iftDestroyGraphicalContext(gc);

    iftDestroyImage(&rend[0]);
    iftDestroyImage(&rend[1]);
    iftDestroyImage(&rend[2]);
    iftDestroyImage(&img);

    /* ---------------------------------------------------------- */

    MemDinFinal = iftMemoryUsed();
    if (MemDinInicial!=MemDinFinal)
        printf("\n\nDinamic memory was not completely deallocated (%lu, %lu)\n",
               MemDinInicial,MemDinFinal);

    return(0);
}
