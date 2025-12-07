#include "ift.h"

Scene *Thres(Scene *scn, int lower, int higher)
{
  Scene *bin=NULL;
  int p,n;

  bin = CreateScene(scn->xsize,scn->ysize,scn->zsize);
  bin->dx = scn->dx;
  bin->dy = scn->dy;
  bin->dz = scn->dz;
  n = scn->xsize*scn->ysize*scn->zsize;
  for (p=0; p < n; p++)
    if ((scn->data[p] >= lower)&&(scn->data[p] <= higher))
      bin->data[p]=255;
  bin->maxval = 255;
  return(bin);
}

int Moda(Curve *hist)
{
  int i,moda=0,max=INT_MIN;

  for (i=0; i < hist->n; i++) 
    if (hist->Y[i] > max){
      max    = hist->Y[i];
      moda   = hist->X[i];
    }
  return(moda);
}

void FindKMaxima(Curve *hist, int *max, int k)
{
  int i,j,n,maxval,moda;
  Queue *Q=NULL;
  Curve *shist=CreateCurve(hist->n),*chist=CreateCurve(hist->n);
  int *val=(int *)calloc(k,sizeof(int)),K;
  char valid;

  moda   = Moda(hist);
  maxval = hist->Y[moda];
  K      = 10;

  for (i=1; i < hist->n-1; i++) {
    if ((hist->Y[i] < hist->Y[i+1])||
	(hist->Y[i] < hist->Y[i-1]))
      chist->Y[i]=0.0;
    else
      chist->Y[i]=hist->Y[i];
    chist->X[i]=hist->X[i];
  }

  /* Sort values of the histogram in decreasing order of Y */

  Q = CreateQueue(maxval+1,chist->n);
  for(i=0; i < chist->n; i++)
    InsertQueue(Q,(int)chist->Y[i],i);

  for (i=0; i < shist->n-1; i++) {
    shist->Y[i]=0.0;
  }

  j = chist->n-1;
  while(!EmptyQueue(Q)) {
    i = RemoveQueue(Q);
    shist->Y[j] = chist->Y[i];
    shist->X[j] = chist->X[i];
    j--;
  }
  DestroyQueue(&Q);


  /* Copy the k most significant maxima, separated by at least K
     bins */

  val[0] = shist->X[0];
  j      = 1;
  for (i=1; (i < shist->n)&&(j < k); i++){
    valid = 1;
    for (n=0; n <= j; n++){
      if ((abs((int)shist->X[i]-val[n]) < K)||
	  (shist->Y[i]==0.0))
     {
	valid=0;
	break;
      }
    }
    if (valid){
      val[j] = (int)shist->X[i];
      j++;
    }
  }
  
  DestroyCurve(&shist);
  DestroyCurve(&chist);

  /* Sort the k maxima in decreasing order */

  Q = CreateQueue(hist->n+1,k);
  for(i=0; i < k; i++)
    InsertQueue(Q,val[i],i);

  j = k-1;
  while(!EmptyQueue(Q)) {
    i = RemoveQueue(Q);
    max[j] = val[i];
    j--;
  }
  DestroyQueue(&Q);
  free(val);
}

void FindKMinima(Curve *hist, int *min, int k)
{
  int i,j,n,maxval,moda;
  Queue *Q=NULL;
  Curve *shist=CreateCurve(hist->n),*chist=CreateCurve(hist->n);
  int *val=(int *)calloc(k,sizeof(int)),K;
  char valid;

  moda   = Moda(hist);
  maxval = hist->Y[moda];
  K      = 10;

  for (i=1; i < hist->n-1; i++) {
    if ((hist->Y[i] > hist->Y[i+1])||
	(hist->Y[i] > hist->Y[i-1]))
      chist->Y[i]=0.0;
    else
      chist->Y[i]=maxval-hist->Y[i];
    chist->X[i]=hist->X[i];
  }

  /* Sort values of the histogram in decreasing order of Y */

  Q = CreateQueue(maxval+1,chist->n);
  for(i=0; i < chist->n; i++)
    InsertQueue(Q,(int)chist->Y[i],i);

  for (i=0; i < shist->n-1; i++) {
    shist->Y[i]=0.0;
  }

  j = chist->n-1;
  while(!EmptyQueue(Q)) {
    i = RemoveQueue(Q);
    shist->Y[j] = chist->Y[i];
    shist->X[j] = chist->X[i];
    j--;
  }
  DestroyQueue(&Q);


  /* Copy the k most significant maxima, separated by at least K
     bins */

  val[0] = shist->X[0];
  j      = 1;
  for (i=1; (i < shist->n)&&(j < k); i++){
    valid = 1;
    for (n=0; n <= j; n++){
      if ((abs((int)shist->X[i]-val[n]) < K)||
	  (shist->Y[i]==0.0))
     {
	valid=0;
	break;
      }
    }
    if (valid){
      val[j] = (int)shist->X[i];
      j++;
    }
  }
  
  DestroyCurve(&shist);
  DestroyCurve(&chist);

  /* Sort the k maxima in increasing order */

  Q = CreateQueue(hist->n+1,k);
  for(i=0; i < k; i++)
    InsertQueue(Q,val[i],i);

  j = 0;
  while(!EmptyQueue(Q)) {
    i = RemoveQueue(Q);
    min[j] = val[i];
    j++;
  }
  DestroyQueue(&Q);
  free(val);
}

Curve *Smooth(Curve *hist)
{
  Curve *fhist=CreateCurve(hist->n);
  int i,j,l;
  double k[21];

  for (i=0; i < fhist->n-1; i++) {
    fhist->Y[i]=0.0;
  }

  for (j=-10; j <= 10; j++){ 
    k[j+10] = exp(-((double)j*(double)j)/64.0);
  }

  for (i=10; i < hist->n-10; i++) {
    for (j=i-10,l=0; j <= i+10; j++,l++)
      fhist->Y[i] += (hist->Y[j]*k[l]);
    fhist->X[i] = hist->X[i];
  }
  for (i=0; i < 10; i++) {
    fhist->Y[i] = hist->Y[i];
    fhist->X[i] = hist->X[i];
  }
  for (i=hist->n-10; i < hist->n; i++) {
    fhist->Y[i] = hist->Y[i];
    fhist->X[i] = hist->X[i];
  }

  return(fhist);
}

Curve *RegMin1(Curve *hist, int w)
{
  Curve *rmin=NULL,*cost=NULL,*pred=NULL;
  Queue *Q=NULL;
  int i,j,moda;

  rmin = CreateCurve(hist->n);
  cost = CreateCurve(hist->n);
  pred = CreateCurve(hist->n);

  
  moda = Moda(hist);
  Q = CreateQueue((int)hist->Y[moda]+1,hist->n);
  for (i=0; i < hist->n; i++) {
    rmin->X[i]=hist->X[i];
    cost->X[i]=hist->X[i];
    pred->X[i]=hist->X[i];
    rmin->Y[i]=0;
    cost->Y[i]=hist->Y[i];
    pred->Y[i]=i;
    InsertQueue(Q,(int)cost->Y[i],i);
  }

  while (!EmptyQueue(Q)){
    j = RemoveQueue(Q);
    if (pred->Y[j]==j)
      rmin->Y[j]=1;
    for (i=j-w; i <= j+w; i++) {
      if ((i >= 0)&&(i < hist->n)){
	if (hist->Y[i] >= hist->Y[j]){
	  if (cost->Y[i] > cost->Y[j]){
	    UpdateQueue(Q,i,(int)cost->Y[i],(int)cost->Y[j]);
	    cost->Y[i] = cost->Y[j];
	    pred->Y[i] = j;
	  }
	}
      }
    }
  }
  DestroyQueue(&Q);
  DestroyCurve(&cost);
  DestroyCurve(&pred);

  return(rmin);
}

Curve *HDomes1(Curve *hist, int H)
{
  Curve *marker=CreateCurve(hist->n);
  int i,j,k,max=INT_MIN,tmp;
  Queue *Q=NULL;

  for (i=0; i < hist->n; i++) {
    marker->Y[i] = MAX((int)hist->Y[i]-H,0);
    marker->X[i] = hist->X[i];
    
    if ((int)marker->Y[i] > max)
      max = (int)marker->Y[i];
  }

  Q = CreateQueue(max+1,hist->n);
  for (i=0; i < hist->n; i++) {    
    InsertQueue(Q,max-(int)marker->Y[i],i);
  }  
  while(!EmptyQueue(Q)) {
    j=RemoveQueue(Q);

    for (i=-1; i <= 1; i++){
      k = i+j;
      if ((k >= 0)&&(k < hist->n)){ 
	if (marker->Y[j] > marker->Y[k]){
	  tmp = MIN((int)marker->Y[j],(int)hist->Y[k]);
	  if (tmp > (int)marker->Y[k]){
	    UpdateQueue(Q,k,max - (int)marker->Y[k],max - tmp);
	    marker->Y[k] = tmp;
	  }
	}
      }
    }
  }
  DestroyQueue(&Q);
  for (i=0; i < marker->n; i++) {
    marker->Y[i] = hist->Y[i]-marker->Y[i];
  }

  return(marker);
}

Scene *CloseHoles25(Scene *scn)
{
  Scene *zscn=CreateScene(scn->xsize,scn->ysize,scn->zsize);
  int z;
  Image *tmp1=NULL,*tmp2=NULL;
  
  /* Along Z */

  for (z = 0; z < scn->zsize; z++) {
    tmp1 = GetSlice(scn,z);
    tmp2 = CloseHoles(tmp1);
    PutSlice(tmp2,zscn,z);
    DestroyImage(&tmp1);
    DestroyImage(&tmp2);
  }
  return(zscn);
}


int main(int argc, char **argv)
{
  timer tic,toc;
  FILE *ftime=fopen("time.txt","w");
  Scene *scn=NULL,*bin=NULL, *tmp=NULL;
  Curve *hist1=NULL,*hist2=NULL;
  int lower=0,higher,max[5];
  Shell *sh=NULL;
  AdjRel3 *A=NULL;
 
  /*--------------------------------------------------------*/

  void *trash = malloc(1);                 
  struct mallinfo info;   
  int MemDinInicial, MemDinFinal;
  free(trash); 
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc != 3) {
    fprintf(stderr,"usage: boneshell <inputfile> <outputfile>\n");
    exit(-1);
  }

  /* Read scene of the head */

  gettimeofday(&tic,NULL);  
  scn  = ReadScene(argv[1]);
  
  /* Compute Filtered Histogram */

  hist1   = Histogram3(scn);
  hist2   = Smooth(hist1);
  DestroyCurve(&hist1);
  hist1   = HDomes1(hist2, (int)(0.2*hist2->Y[Moda(hist2)]));
  WriteCurve(hist1,"hist.txt");
  DestroyCurve(&hist2);
  FindKMaxima(hist1,max,5);   
  DestroyCurve(&hist1);

  /* Compute threholding, filtering and get bone mask */

   higher = MaximumValue3(scn); // bone
   lower  = 1.05*max[0];
   fprintf(stdout,"lower %d higher %d\n",lower,higher);

   tmp    = Threshold3(scn,lower,higher);
   A = Spheric(1.8);
   bin  = MedianFilter3(tmp,A);
   DestroyScene(&tmp);
   DestroyAdjRel3(&A);
   tmp    = Mult3(scn,bin);
   DestroyScene(&scn);
   sh     = Object2Shell(bin,5);
   SetShellNormal (sh, tmp, Gradient3);
   DestroyScene(&tmp);
   SetShellLabel(sh,bin); 
   gettimeofday(&toc,NULL);
   fprintf(ftime,"BoneShell in %f milliseconds\n",((toc.tv_sec-tic.tv_sec)*1000.0 + (toc.tv_usec-tic.tv_usec)*0.001));


   WriteShell(sh,argv[2]);
   DestroyShell(&sh);
   DestroyScene(&bin);

  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);   

  fclose(ftime);
  return(0);
}
