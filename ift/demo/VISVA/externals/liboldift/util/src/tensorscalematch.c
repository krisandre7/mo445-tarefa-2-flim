#include "ift.h"

int main(int argc, char **argv){
  timer tic,toc;
  Image *img,*img2;
  Image *thick;
  CImage *cimg;
  TensorScale *ts,*ts2;
  char filename[200];
  int  offset;//,i;
  float totaltime;
  float *hist1,*hist2;
  float score;
  
  if (argc != 3) {
    fprintf(stderr,"usage: tsmatch <in:pgmfile> <in2:pgmfile>\n");
    exit(-1);
  }

  // Le primeira imagem.
  sprintf(filename,"%s",argv[1]);
  img = ReadImage(filename);

  // Le segunda imagem.
  sprintf(filename,"%s",argv[2]);
  img2 = ReadImage(filename);

  // Calcula o tensorscale para a primeira imagem.
  gettimeofday(&tic,NULL);
  ts = CreateBinaryTensorScale(img, 24);
  gettimeofday(&toc,NULL);
  totaltime = (toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001;
  printf("\nTime1: %f ms\n",totaltime);
  // Gera histograma da imagem de orientacao.
  hist1 = TSOrientationHistogram(ts);

  thick = ConvertDImage2Image(ts->thickness);
  WriteImage(thick,"thickness1.pgm");
  DestroyImage(&thick);

  // Calcula o tensorscale para a segunda imagem.
  gettimeofday(&tic,NULL);
  ts2 = CreateBinaryTensorScale(img2, 24); //24, 16
  gettimeofday(&toc,NULL);
  totaltime = (toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001;
  printf("\nTime2: %f ms\n",totaltime);
  // Gera histograma da imagem de orientacao.
  hist2 = TSOrientationHistogram(ts2);

  thick = ConvertDImage2Image(ts2->thickness);
  WriteImage(thick,"thickness2.pgm");
  DestroyImage(&thick);

  //for(i=0;i<256;i++){
  //  printf("i: %d, hist1: %f, hist2: %f\n",i,hist1[i],hist2[i]);
  //}

  // Gera imagem colorida com os atributos de thickness, anisotropy e 
  // orientation mapeados no espaco de cores HSV.
  cimg = ConvertTS2CImage(ts);
  WriteCImage(cimg,"ts1.pgm");
  DestroyCImage(&cimg);

  // Gera imagem colorida com os atributos de thickness, anisotropy e 
  // orientation mapeados no espaco de cores HSV.
  cimg = ConvertTS2CImage(ts2);
  WriteCImage(cimg,"ts2.pgm");
  DestroyCImage(&cimg);

  // Compara os histogramas de orientacao das duas imagens.
  score = TSHistogramMatch(hist1, hist2, &offset);
  printf("\nScore: %f, offset: %d\n",score,offset);
  cimg = TSShowHistograms(hist1, hist2, offset);
  WriteCImage(cimg,"plot.pgm");
  DestroyCImage(&cimg);

  DestroyImage(&img);
  DestroyTensorScale(&ts);

  DestroyImage(&img2);
  DestroyTensorScale(&ts2);

  return 0;
}
 
