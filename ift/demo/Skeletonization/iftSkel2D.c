#include "ift.h"

inline int value(iftImage* img,int x,int y)
{
    return img->val[y*img->xsize+x];
}

void float2rgb(float value,float* R,float* G,float* B)	//simple color-coding routine
{
      float dx=0.8f;

    value = (6-2*dx)*value+dx;
    *R = iftMax(0.0f,(3-(float)fabs(value-4)-(float)fabs(value-5))/2);
    *G = iftMax(0.0f,(4-(float)fabs(value-2)-(float)fabs(value-4))/2);
    *B = iftMax(0.0f,(3-(float)fabs(value-1)-(float)fabs(value-2))/2);
}


void saveSkeleton(iftFImage* img,iftImage* bimg,const char* name)
{
    FILE* fp = fopen(name,"wb");

    float maxval = iftFMaximumValue(img);
    float minval = iftFMinimumValue(img);

    fprintf(fp,"P6 %d %d 255\n",img->xsize,img->ysize);

    int size = img->xsize*img->ysize;
    for(float* p=img->val;p<img->val+size;++p)
    {
        unsigned char d[3];

        float v = *p;

        if (v<1)                        //Not on skeleton: draw black if on shape boundary, else white
        {
            int offs = p - img->val;
            int x    = offs % img->xsize;
            int y    = offs / img->xsize;
            if (x>0 && y>0 && x<img->xsize-1 && y<img->ysize-1)
            {                           //Test if on shape boundary only if not on image borders
                int cp = value(bimg,x,y);
                if (cp!=value(bimg,x-1,y) || cp!=value(bimg,x+1,y) || cp!=value(bimg,x,y-1) ||cp!=value(bimg,x,y+1))
                  d[0]=d[1]=d[2]=0;
                else
                  d[0]=d[1]=d[2]=255;
            }
            else
                d[0]=d[1]=d[2]=255;    //Not on shape boundary
        }
        else                            //On skeleton: color-code importance value
        {
            v = (v-minval)/(maxval-minval);

            float r,g,b;
            float2rgb(v,&r,&g,&b);


            d[0] = (unsigned char)(int)(r*255);
            d[1] = (unsigned char)(int)(g*255);
            d[2] = (unsigned char)(int)(b*255);
        }

        fwrite(d,1,3,fp);
    }

    fclose(fp);
}


int main(int argc, char *argv[])
{
  iftImage       *bin=NULL, *skel=NULL, *skel_255=NULL, *border=NULL;
  iftAdjRel      *A=NULL,*B=NULL;
  iftFImage      *msskel=NULL;
  timer          *t1=NULL,*t2=NULL;

  /*--------------------------------------------------------*/

  size_t       MemDinInitial, MemDinFinal;

  MemDinInitial = iftMemoryUsed();

  /*--------------------------------------------------------*/


  if (argc!=2)
    iftError("Usage: iftSkel2D <binary.[pgm,png,jpg,tif]>","main");

  A      = iftCircular(sqrtf(2.0));

  bin    = iftReadImageByExt(argv[1]);
  
  if (!iftIsBinaryImage(bin)){
    iftImage *aux= iftThreshold(bin,127,255,255);
    iftDestroyImage(&bin);
    bin = aux;
  }

  t1     = iftTic();  
  msskel   = iftMSSkel2D(bin,A,IFT_INTERIOR, NULL, NULL);
  t2     = iftToc();
  fprintf(stdout,"skeletonization in %f ms\n",iftCompTime(t1,t2));

  // you may use iftColorMSSkel2D instead, but it will compute the
  // multiscale skeletons again.
  saveSkeleton(msskel,bin,"msskel.png");
  
  iftDestroyAdjRel(&A);

  float T;
  printf("Enter a threshold from 0.01 to 99.0 and see image skel.png\n");
  printf("or enter 0 to terminate the program.\n");
  scanf("%f",&T);
  iftColor RGB,YCbCr;
  RGB.val[0]=255;
  RGB.val[1]=50;
  RGB.val[2]=50;
  YCbCr = iftRGBtoYCbCr(RGB,255);
  A = iftCircular(3.0);
  B = iftCircular(1.0);
  iftAdjRel *C = iftCircular(0.0);
  while (T!=0.0)
  {
    skel     = iftFThreshold(msskel,T,100.0,1);
    skel_255 = iftBinaryFrom1To255(skel);
    border   = iftFindAndLabelObjectBorders(bin, B);
    iftDrawObject(skel_255,border, YCbCr, C);

    RGB.val[0]=0;
    RGB.val[1]=255;
    RGB.val[2]=255;
    YCbCr     = iftRGBtoYCbCr(RGB,255);

    iftImage *term = iftTerminalPoints2D(skel);
    for (int p=0; p < term->n; p++)
      if (term->val[p])
        iftDrawPoint(skel_255,iftGetVoxelCoord(term,p),YCbCr,B,255);


    iftWriteImageP6(skel_255,"skel.png");
    iftDestroyImage(&skel);
    iftDestroyImage(&skel_255);
    iftDestroyImage(&border);
    printf("Enter a threshold from 0.01 to 99.0 and see image skel.png\n");
    printf("or enter 0 to terminate the program.\n");
    scanf("%f",&T);
  }

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
  iftDestroyAdjRel(&C);
  iftDestroyImage(&bin);

  /* ---------------------------------------------------------- */


  MemDinFinal = iftMemoryUsed();
  iftVerifyMemory(MemDinInitial,MemDinFinal);

  return(0);
}
