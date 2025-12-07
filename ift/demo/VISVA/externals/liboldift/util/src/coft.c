#include "ift.h"


// Object-based similarity

Image *ObjectSimilarity(Image *img)
{
    real    dist,gx,gy;
    int     i,p,q,n=img->ncols*img->nrows;
    Pixel   u,v;
    AdjRel *A=Circular(1.5);
    real   *md=AllocRealArray(A->n);

    Image* simil = CreateImage(img->ncols, img->nrows);

    int Imax = MaximumValue(img);

    for (i=1; i < A->n; i++)
        md[i]=sqrt(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]);

    for (p=0; p < n; p++)
    {
        u.x = p%img->ncols;
        u.y = p/img->ncols;

        gx = gy = 0.0;

        for (i=1; i < A->n; i++)
        {
            v.x = u.x + A->dx[i];
            v.y = u.y + A->dy[i];
            if (ValidPixel(img,v.x,v.y))
            {
                q    = v.x + img->tbrow[v.y];
                dist = ((float)img->val[q]-(float)img->val[p])/Imax;

                gx  += dist*A->dx[i]/md[i];
                gy  += dist*A->dy[i]/md[i];
            }
        }
        simil->val[p]=(int)1000*sqrt(gx*gx + gy*gy);
    }

    Imax = MaximumValue(simil);
    for (p=0; p < n; p++)
      simil->val[p]=Imax-simil->val[p];

    free(md);
    DestroyAdjRel(&A);

    return(simil);
}

// Image-based similarity

Image *ImageSimilarity(Features *f)
{
    real    dist,gx,gy,mag;
    int     j,i,p,q,n=f->ncols*f->nrows,Imax;
    Pixel   u,v;
    AdjRel *A=Circular(1.5);
    real   *md=AllocRealArray(A->n);

    Image* simil = CreateImage(f->ncols, f->nrows);

    for (i=1; i < A->n; i++)
        md[i]=sqrt(A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i]);

    for (p=0; p < n; p++)
    {
        u.x = p%f->ncols;
        u.y = p/f->ncols;

        float max_mag = -FLT_MAX;
        for (j=0; j<f->nfeats; j++)
        {
            gx = gy = 0.0;
            for (i=1; i < A->n; i++)
            {
                v.x = u.x + A->dx[i];
                v.y = u.y + A->dy[i];
                if ((v.x>=0 && v.x<f->ncols) && (v.y>=0 && v.y<f->nrows))
                {
                    q    = v.x + v.y*f->ncols;
                    dist = (f->elem[q].feat[j]-f->elem[p].feat[j]);
                    gx  += dist*A->dx[i]/md[i];
                    gy  += dist*A->dy[i]/md[i];
                }
            }
            mag = sqrt(gx*gx + gy*gy);

            if (mag > max_mag)
                max_mag = mag;
        }
        simil->val[p] = (int)1000*max_mag;
    }

    Imax = MaximumValue(simil);
    for (p=0; p < n; p++)
      simil->val[p]=Imax-simil->val[p];

    free(md);
    DestroyAdjRel(&A);

    return(simil);
}

// Linear combination of similarities

Image* CombineSimilarities(Image *objsim, Image *imgsim, float wobj)
{
  int p;
  Image *simil = CreateImage(objsim->ncols, objsim->nrows);

  for(p = 0; p < objsim->ncols*objsim->nrows; p++)
    simil->val[p] = wobj*objsim->val[p] + (1-wobj)*imgsim->val[p];

  return simil;
}

Subgraph* MySubgraphFromSeeds(Features* f, Set* S, int nsamples)
{

    Subgraph *sg=NULL;
    Set *aux;
    int n,i,j;
    int *used,*elem;

    n = GetSetSize(S);

    if (n < nsamples){ 
      printf("Using all %d samples\n",n);
      sg = CreateSubgraph(n);

      i = 0;
      aux = S;
      while (aux != NULL)
	{
	  sg->node[i].pixel = aux->elem;
	  aux = aux->next;
	  i++;
	}
    }else{

      //----- Get all seeds -------------------

      used = AllocIntArray(n);
      elem = AllocIntArray(n);
      
      i = 0;
      aux = S;
      while (aux != NULL)
	{
	  elem[i] = aux->elem;
	  aux = aux->next;
	  i++;
	}
      
      //--------CreateSubGraph by random sampling------------
      
      sg = CreateSubgraph(nsamples);
      
      i    = 0;
      while( i < nsamples ) {
	j = RandomInteger( 0, n - 1 );
	if ( used[ j ] == 0 ) {
	  sg->node[ i ].pixel  = elem[j];
	  used[ j ] = 1;
	  i++;
	}
      }
      
      free( used );
      free( elem );
    }

    SetSubgraphFeatures(sg,f);

    return sg;
}

Image *MyPropagatePDF(Subgraph *sg, Features *f)
{
    int     i,p,k;
    Image  *pdf=CreateImage(f->ncols,f->nrows);
    float   dist;

    for (p=0; p < f->nelems; p++)
    {
      for (i = 0; i < sg->nnodes; i++){
	k = sg->ordered_list_of_nodes[i];
	dist = opf_ArcWeight(sg->node[k].feat,f->elem[p].feat,sg->nfeats);

	if (dist <= sg->node[k].radius){
	  pdf->val[p]=(int)(sg->node[k].pathval*exp(-dist/sg->K));
	  break;
	}
      }
    }
    return(pdf);
}

// Using Linear Convolution with Gaussian filters

Features *GaussImageFeats(Image *img, int nscales)
{
    Features *f=CreateFeatures(img->ncols, img->nrows, nscales);
    AdjRel   *A=NULL;
    int       s,i,p,q;
    Pixel     u,v;
    float    *w,d,K1,K2,sigma2,val,sum;

    f->Imax = MaximumValue(img);

    for (s=1; s <= nscales; s=s+1)
    {
        A  = Circular(s);
        w  = AllocFloatArray(A->n);
        sigma2 = (s/3.0)*(s/3.0);
        K1     =  2.0*sigma2;
        K2     = 1.0/sqrt(2.0*PI*sigma2);
        //compute kernel coefficients
	sum = 0.0;
        for (i=0; i < A->n; i++)
        {
            d    = A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i];
            w[i] = K2 * exp(-d/K1); // Gaussian
	    sum += w[i];
        }

        // Convolution

        for (p=0; p < f->nelems; p++)
        {
            u.x = p%f->ncols;
            u.y = p/f->ncols;
            val = 0.0;
            for (i=0; i < A->n; i++)
            {
                v.x = u.x + A->dx[i];
                v.y = u.y + A->dy[i];
                if (ValidPixel(img,v.x,v.y))
                {
                    q   = v.x + img->tbrow[v.y];
                    val += (float)img->val[q]*w[i];
                }
            }
            f->elem[p].feat[s-1]=(int)val/(sum*f->Imax);
        }
        free(w);

        DestroyAdjRel(&A);
    }
    return(f);
}


Features *GaussCImageFeats(CImage *cimg, int nscales)
{
    Features *f=NULL;
    AdjRel   *A=NULL;
    int       s,i,j,p,q;
    Pixel     u,v;
    Image    *img1;
    float    *w,d,K,sigma2,val,sum;

    f = CreateFeatures(cimg->C[0]->ncols, cimg->C[0]->nrows, 3*nscales);

    f->Imax = MAX(MAX(MaximumValue(cimg->C[0]),MaximumValue(cimg->C[1])),MaximumValue(cimg->C[2]));

    for (j=0; j < 3; j=j+1)
    {
        img1 = cimg->C[j];
        for (s=1; s <= nscales; s=s+1)
        {
            A  = Circular(s);
            w  = AllocFloatArray(A->n);
            sigma2 = (s/3.0)*(s/3.0);
            K      =  2.0*sigma2;

            //compute kernel coefficients
	    sum = 0.0;
            for (i=0; i < A->n; i++)
            {
                d    = A->dx[i]*A->dx[i]+A->dy[i]*A->dy[i];
                w[i] = 1.0/sqrt(2.0*PI*sigma2) * exp(-d/K); // Gaussian
		sum += w[i];
	    }

            // Convolution

            for (p=0; p < f->nelems; p++)
            {
                u.x = p%f->ncols;
                u.y = p/f->ncols;
                val = 0.0;
                for (i=0; i < A->n; i++)
                {
                    v.x = u.x + A->dx[i];
                    v.y = u.y + A->dy[i];
                    if (ValidPixel(img1,v.x,v.y))
                    {
                        q   = v.x + img1->tbrow[v.y];
                        val += (float)img1->val[q]*w[i];
                    }
                }
                f->elem[p].feat[s-1+(j*nscales)] = val/(sum*f->Imax);
            }
            free(w);

            // Copy features and reinitialize images

            DestroyAdjRel(&A);
        }
    }
    return(f);
}

void ReadSeeds(char *filename, Set **Obj, Set **Bkg)
{
  FILE *fp=fopen(filename,"r");
  int i,x,y,l,mk,nseeds, ncols, nrows;

  if(fscanf(fp,"%d %d %d",&nseeds, &ncols, &nrows)!=0);

  for (i=0; i < nseeds; i++){
    if(fscanf(fp,"%d %d %d %d",&x,&y,&mk,&l)!=0);
    if (l==0)
      InsertSet(Bkg, x + ncols*y);
    else
      InsertSet(Obj, x + ncols*y);
  }
  fclose(fp);
}

Image *COFTsegmentation(Image *img, Set *Obj, Set *Bkg)
{
  AdjRel *A=NULL;
  GQueue *Q=NULL;
  Image  *pathvalue=NULL,*label=NULL;
  Pixel   u,v;
  int     i,p,q,n,tmp,Vmax=MaximumValue(img);
  Set    *S;

  pathvalue  = CreateImage(img->ncols,img->nrows);
  label = CreateImage(img->ncols,img->nrows);
  n     = img->ncols*img->nrows;
  Q     = CreateGQueue(Vmax+1,n,pathvalue->val);
  SetRemovalPolicy(Q,MAXVALUE);      
  A     = Circular(1.5);

  /* Trivial path initialization */

  for (p=0; p < n; p++){
    pathvalue->val[p] =INT_MIN;
  }
  S = Obj;
  while(S != NULL){
    p=S->elem;
    label->val[p]=1;
    pathvalue->val[p]=img->val[p];
    InsertGQueue(&Q,p);
    S = S->next;
  }
  S = Bkg;
  while(S != NULL){
    p=S->elem;
    label->val[p]=0;
    pathvalue->val[p]=img->val[p];
    InsertGQueue(&Q,p);
    S = S->next;
  }

  /* Path propagation */

  while (!EmptyGQueue(Q)){
    p   = RemoveGQueue(Q);
    u.x = p%img->ncols;
    u.y = p/img->ncols;
    for (i=1; i < A->n; i++) {
      v.x = u.x + A->dx[i];
      v.y = u.y + A->dy[i];
      if (ValidPixel(img,v.x,v.y)){
	q   = v.x + img->tbrow[v.y];
	if (pathvalue->val[p] > pathvalue->val[q]){

	  tmp = MIN(pathvalue->val[p] , img->val[q]);
	  if (tmp > pathvalue->val[q]){
	    if (pathvalue->val[q]!=INT_MIN)
	      RemoveGQueueElem(Q,q);
	    pathvalue->val[q] =tmp;
	    label->val[q]=label->val[p];
	    InsertGQueue(&Q,q);
	  }
	}

      }
    }
  }


  DestroyGQueue(&Q);
  DestroyImage(&pathvalue);
  DestroyAdjRel(&A);

  return(label);
}

Set *GetOutlierSet(Subgraph *sg, Set *S, Features *f)
{
  int     i,k,outlier;
  float   dist;
  Set    *Saux=S,*newSet=NULL;

  while (Saux!=NULL) {
    outlier=1;
    for (i = 0; i < sg->nnodes; i++){
      k = sg->ordered_list_of_nodes[i];
      dist = opf_ArcWeight(sg->node[k].feat,f->elem[S->elem].feat,f->nfeats);      
      if (dist <= sg->node[k].radius){
	outlier=0;
	break;
      }
    }
    if (outlier){
      InsertSet(&newSet,Saux->elem);
    }
    Saux = Saux->next;
  }
  
  return(newSet);
}

int main(int argc, char **argv)
{
  timer    *t1=NULL,*t2=NULL;
  Image    *img=NULL,*pdfObj=NULL,*pdfBkg=NULL,*label=NULL;
  CImage   *cimg=NULL;
  Features *feat=NULL;
  Subgraph *sgObj=NULL,*sgBkg=NULL;
  Set      *Obj=NULL,*Bkg=NULL,*trainBkg=NULL,*trainObj=NULL;
  char      outfile[100];
  char     *file_noext;

  /*--------------------------------------------------------*/

  void *trash = malloc(1);
  struct mallinfo info;
  int MemDinInicial, MemDinFinal;
  free(trash);
  info = mallinfo();
  MemDinInicial = info.uordblks;

  /*--------------------------------------------------------*/

  if (argc!=3){
    fprintf(stderr,"Usage: coft <image.ppm>  <seeds.txt>\n");
    fprintf(stderr,"image.pgm: image to be classified\n");
    fprintf(stderr,"seeds.txt: seed pixels\n");
    exit(-1);
  }

  char *ext = strrchr(argv[1],'.');

  if(strcmp(ext,".ppm")==0){
    cimg   = ReadCImage(argv[1]);
    img    = ift_CopyImage(cimg->C[1]);

    Features *gaussfeats   = GaussCImageFeats(cimg, 2);
    int p,j;
    //converting features from [0,1] to [0,255]
    for(p = 0; p < gaussfeats->nelems; p++)
      for(j = 0; j < gaussfeats->nfeats; j++)
	gaussfeats->elem[p].feat[j] *= gaussfeats->Imax;
    
    feat = LabFeats(gaussfeats);
  
  //converting features from [0,255] to [0,1]
    for(p = 0; p < feat->nelems; p++){
      for(j = 0; j < feat->nfeats; j++)
	feat->elem[p].feat[j] /= feat->Imax;
    }
    DestroyFeatures(&gaussfeats);

  }else{
    printf("Invalid image format\n");
    exit(1);
  }

  file_noext = strtok(argv[1],".");

  ReadSeeds(argv[2],&Obj,&Bkg);

  t1 = Tic();

  /* OPF-based binary classification of the image */

  Image *imgSim=ImageSimilarity(feat),*simil=NULL,*osim=NULL,*bsim=NULL;

  WriteImage(imgSim,"imgsimil.pgm");

  sgObj = MySubgraphFromSeeds(feat,Obj,500);
  opf_BestkMinCut(sgObj,(int)(0.10*sgObj->nnodes),(int)(0.40*sgObj->nnodes));
  opf_OPFClustering(sgObj);

  sgBkg = MySubgraphFromSeeds(feat,Bkg,500);
  opf_BestkMinCut(sgBkg,(int)(0.10*sgBkg->nnodes),(int)(0.40*sgBkg->nnodes));
  opf_OPFClustering(sgBkg);

  trainObj = GetOutlierSet(sgBkg,Obj,feat);
  trainBkg = GetOutlierSet(sgObj,Bkg,feat);

  int p,n=imgSim->ncols*imgSim->nrows;
  Image *objSim=CreateImage(imgSim->ncols,imgSim->nrows);
  for (p=0; p < n; p++) {
    objSim->val[p] = INT_MAX;
  }


  if (trainBkg!=NULL) {
    DestroySubgraph(&sgBkg);
    sgBkg = MySubgraphFromSeeds(feat,trainBkg,500);
    printf("recomputing background forest\n");
    opf_BestkMinCut(sgBkg,(int)(0.10*sgBkg->nnodes),(int)(0.40*sgBkg->nnodes));
    opf_OPFClustering(sgBkg);
    pdfBkg=MyPropagatePDF(sgBkg,feat);
    WriteImage(pdfBkg,"pdfbkg.pgm");
    bsim=ObjectSimilarity(pdfBkg);
    for (p=0; p < n; p++) {
      objSim->val[p] = MIN(objSim->val[p],bsim->val[p]);
    }
    DestroyImage(&bsim);
    DestroyImage(&pdfBkg);
  }  

  
  if (trainObj!=NULL) {
    DestroySubgraph(&sgObj);
    sgObj = MySubgraphFromSeeds(feat,trainObj,500);
    printf("recomputing object forest\n");
    opf_BestkMinCut(sgObj,(int)(0.10*sgObj->nnodes),(int)(0.40*sgObj->nnodes));
    opf_OPFClustering(sgObj);
    pdfObj=MyPropagatePDF(sgObj,feat);
    WriteImage(pdfObj,"pdfobj.pgm");
    osim=ObjectSimilarity(pdfObj);
    for (p=0; p < n; p++) {
      objSim->val[p] = MIN(objSim->val[p],osim->val[p]);
    }
    DestroyImage(&osim);
    DestroyImage(&pdfObj);
  }
  

  if ((trainBkg!=NULL)||(trainObj!=NULL)) {

    WriteImage(objSim,"objsimil.pgm");

    simil=CombineSimilarities(objSim,imgSim,0.7);

    WriteImage(simil,"combsimil.pgm");

  
  }else{

    simil=ift_CopyImage(imgSim);

  }

  DestroyImage(&objSim);

  label = COFTsegmentation(simil,Obj,Bkg);

  t2 = Toc();

  fprintf(stdout,"coft time in %f ms\n",CTime(t1,t2));
  
  int q,i;
  Pixel u,v;
  AdjRel *A=Circular(1.0);

  /* Object only 

  for (p=0; p < n; p++)
    if (label->val[p]==0){
      cimg->C[0]->val[p]=255;
      cimg->C[1]->val[p]=255;
      cimg->C[2]->val[p]=255;
    }
  */
  
  for (p=0; p < n; p++)
    if (label->val[p]==1){
      u.x = p%label->ncols;
      u.y = p/label->ncols;
      for (i=0; i < A->n; i++) {
	v.x = u.x + A->dx[i];
	v.y = u.y + A->dy[i];
	if (ValidPixel(label,v.x,v.y)){
	  q = v.x + label->tbrow[v.y];
	  if (label->val[q]==0){
	    cimg->C[0]->val[p]=255;
	    cimg->C[1]->val[p]=255;
	    cimg->C[2]->val[p]=0;
	    break;
	  }
	}
      }
    }
  
  DestroyAdjRel(&A);

  // Draw seeds
  /*
  DestroyCImage(&cimg);
  cimg = CreateCImage(simil->ncols, simil->nrows);
  float Imax=MaximumValue(simil);
  for (p=0; p < n; p++){
    cimg->C[0]->val[p]=MIN((int)((float)255.0*simil->val[p]/(float)Imax),255);
    cimg->C[1]->val[p]=MIN((int)((float)255.0*simil->val[p]/(float)Imax),255);
    cimg->C[2]->val[p]=MIN((int)((float)255.0*simil->val[p]/(float)Imax),255);
  }
  */
  while (Obj!=NULL){
    p=RemoveSet(&Obj);
    cimg->C[0]->val[p]=255;
    cimg->C[1]->val[p]=255;
    cimg->C[2]->val[p]=0;
  }
  while (Bkg!=NULL){
    p=RemoveSet(&Bkg);
    cimg->C[0]->val[p]=255;
    cimg->C[1]->val[p]=0;
    cimg->C[2]->val[p]=255;
  }
  
  sprintf(outfile,"%s_result.ppm",strtok(argv[1],"."));
  WriteCImage(cimg,outfile);
  
  DestroyImage(&imgSim);
  DestroyImage(&simil);
  DestroyImage(&img);
  DestroyCImage(&cimg);
  DestroyImage(&label);
  DestroyCImage(&cimg);
  DestroyFeatures(&feat);
  DestroySet(&Obj);
  DestroySet(&Bkg);
  DestroySet(&trainBkg);
  DestroySet(&trainObj);
  DestroySubgraph(&sgObj);
  DestroySubgraph(&sgBkg);



  /* ---------------------------------------------------------- */

  info = mallinfo();
  MemDinFinal = info.uordblks;
  if (MemDinInicial!=MemDinFinal)
    printf("\n\nDinamic memory was not completely deallocated (%d, %d)\n",
	   MemDinInicial,MemDinFinal);

  return(0);
}



