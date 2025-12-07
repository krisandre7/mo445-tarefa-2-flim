
#include "cloudsbrain.h"

//#define APPDEBUG  1
//#define APPTIME   1

//-------------------------------------------
#define NLAYERS  2
#define MAX_WEIGHT 60000

char *bia_dir=NULL;

ArrayList *ComputeWeightPyramid3(Pyramid3 *pyr, 
				 ArrayList *gpyr,
				 MRI_Info info);
ArrayList *ComputeScnGradientPyramid3(Pyramid3 *pyr,
				      MRI_Info info);

void ComputeCloudPyramid3(bia::AdjSeedmap3::AdjSeedmap3 *asmap[NLAYERS],
			  BorderCloud3 *bcloud[NLAYERS],
			  char *basename,
			  Pyramid3 *pyr);
bia::Scene8::Scene8 *SegmObjByMultipleClouds(Pyramid3 *pyr,
					     ArrayList *wpyr,
					     ArrayList *gpyr,
					     MRI_Info info,
					     char *basename,
					     char *bestname,
					     bia::Voxel pos_obj[]);
bia::Scene8::Scene8 *SegmObjBySingleCloud(Pyramid3 *pyr, 
					  ArrayList *wpyr,
					  ArrayList *gpyr,
					  MRI_Info info,
					  char *basename, 
					  real *score,
					  bia::Voxel pos_obj[]);


ArrayList *ComputeWeightPyramid3(Pyramid3 *pyr,
				 ArrayList *gpyr,
				 MRI_Info info){
  ArrayList *wpyr;
  void (*clean)(void**);
  Scene *wobj,*scn;
  bia::Scene16::Scene16 *wobj16;
  ScnGradient *grad;
  MRI_Info lay_info;
  int i,Imax,Wm,p;
  float r=2.0;
#ifdef APPDEBUG
  char filename[512];
#endif

  clean = (void (*)(void**))bia::Scene16::Destroy;
  wpyr = CreateArrayList(NLAYERS);
  SetArrayListCleanFunc(wpyr, clean);

  lay_info = info;
  Wm = MAX_WEIGHT;
  for(i=0; i<NLAYERS; i++){
    scn  = GetPyramid3LayerRef(pyr, i);
    grad = (ScnGradient *)GetArrayListElement(gpyr, i);
    lay_info.COG = VoxelCorrespondencePyr3(pyr,
					   lay_info.COG,
					   MAX(i-1,0), i);
    wobj = FastMRISphericalAccAbsDiff3(scn, r, lay_info);
    MRI_SuppressSkull(wobj, scn, lay_info);
    MRI_SuppressNonBrainBorders(wobj, scn, grad, lay_info);

    Imax = MaximumValue3(wobj);
    SceneNormalize(wobj, 0, Imax, 0, Wm);
    wobj->maxval = Wm;
    //Wm /= 2;

#ifdef APPDEBUG
    if(i==0){
      sprintf(filename,"%s/data/debug/wobj.scn",bia_dir);
      WriteScene(wobj,filename);
    }
#endif

    wobj16 = bia::Scene16::Create(wobj->xsize,
				  wobj->ysize,
				  wobj->zsize);
    wobj16->dx = wobj->dx;
    wobj16->dy = wobj->dy;
    wobj16->dz = wobj->dz;
    wobj16->maxval = (ushort)wobj->maxval;
    for(p=0; p<wobj->n; p++)
      wobj16->data[p] = (ushort)wobj->data[p];
    DestroyScene(&wobj);

    AddArrayListElement(wpyr,
			(void *)wobj16);
  }
  return wpyr;
}



ArrayList *ComputeScnGradientPyramid3(Pyramid3 *pyr,
				      MRI_Info info){
  ArrayList *gpyr;
  void (*clean)(void**);
  ScnGradient *grad;
  Scene *scn;
  float r=1.5;
  int i;
  MRI_Info lay_info;

  clean = (void (*)(void**))DestroyScnGradient;
  gpyr = CreateArrayList(NLAYERS);
  SetArrayListCleanFunc(gpyr, clean);

  lay_info = info;
  for(i=0; i<NLAYERS; i++){
   scn  = GetPyramid3LayerRef(pyr, i);
   lay_info.COG = VoxelCorrespondencePyr3(pyr,
					  lay_info.COG,
					  MAX(i-1,0), i);

   //grad = FastSphericalScnGradient(scn, r);
   grad = MRI_SphericalScnGradient(scn, r, lay_info);
   AddArrayListElement(gpyr,
		       (void *)grad);
  }
  return gpyr;
}


void ComputeCloudPyramid3(bia::AdjSeedmap3::AdjSeedmap3 *asmap[NLAYERS],
			  BorderCloud3 *bcloud[NLAYERS],
			  char *basename,
			  Pyramid3 *pyr){
  char filename[512];
  char voxelsize[512];
  RegionCloud3 *region,*rtmp;
  BorderCloud3 *border,*btmp;
  Scene *scn=NULL;
  float dx,dy,dz;
  int i,Wm;
#ifdef APPDEBUG
  //Scene *aux=NULL;
  //Voxel v;
  //int o;
#endif

  scn = GetPyramid3LayerRef(pyr, 0);

  sprintf(voxelsize,"X%.4fY%.4fZ%.4f",scn->dx,scn->dy,scn->dz);
  ReplaceStringCharacter(voxelsize,'.','_');
  ReplaceStringCharacter(voxelsize,',','_');
  sprintf(filename,"%s/data/models/%s_bcloud_%s.dat",
	  bia_dir,basename,voxelsize);

  if(FileExists(filename)) 
    border = ReadBorderCloud3(filename);

  else{
    sprintf(filename,"%s/data/models/%s_bcloud.dat",
	    bia_dir,basename);
    border = ReadBorderCloud3(filename);
    GetVoxelSizeBorderCloud3(border, &dx,&dy,&dz);

    if(scn->dx!=dx || scn->dy!=dy || scn->dz!=dz){
      btmp = LinearInterpBorderCloud3(border,
				      scn->dx,scn->dy,scn->dz);
      DestroyBorderCloud3(&border);
      border = btmp;
    }
    sprintf(filename,"%s/data/models/%s_bcloud_%s.dat",
	    bia_dir,basename,voxelsize);
    WriteBorderCloud3(border, filename);
  }

  //------------------------------------------------

  sprintf(filename,"%s/data/models/%s_rcloud_%s.dat",
	  bia_dir,basename,voxelsize);

  if(FileExists(filename)) 
    region = ReadRegionCloud3(filename);

  else{
    sprintf(filename,"%s/data/models/%s_rcloud.dat",bia_dir,basename);
    region = ReadRegionCloud3(filename);
    GetVoxelSizeRegionCloud3(region, &dx,&dy,&dz);

    if(scn->dx!=dx || scn->dy!=dy || scn->dz!=dz){
      rtmp = LinearInterpRegionCloud3(region,
				      scn->dx,scn->dy,scn->dz);
      DestroyRegionCloud3(&region);
      region = rtmp;
    }
    sprintf(filename,"%s/data/models/%s_rcloud_%s.dat",
	    bia_dir,basename,voxelsize);
    WriteRegionCloud3(region, filename);
  }

  //------------------------------------------------
  NormalizeBorderCloud3(border,	0,MAX_PROB, 0,MAX_WEIGHT);
  Wm = MAX_WEIGHT;
  //------------------------------------------------
  for(i=0; i<NLAYERS; i++){
    if(i>0){
      rtmp = SubsamplingRegionCloud3(region);
      DestroyRegionCloud3(&region);
      region = rtmp;

      btmp = SubsamplingBorderCloud3(border);
      border = btmp;
      //NormalizeBorderCloud3(border, 0,Wm, 0,Wm/2);
      //Wm /= 2;
    }

    asmap[i]  = bia::AdjSeedmap3::Create(region);
    bcloud[i] = border;

#ifdef APPDEBUG
    /*
    aux = Cloud32Mask(asmap[i]->border[0]);
    v.x = aux->xsize/2;
    v.y = aux->ysize/2;
    v.z = aux->zsize/2;
    for(o=1; o<=asmap[i]->nobjs; o++)
      FastDrawCloud32Scene(asmap[i]->border[o], aux, v, o+1);
    sprintf(filename,"%s/data/debug/bsmap_%s_%02d.scn",bia_dir,basename,i+1);
    WriteScene(aux,filename);
    DestroyScene(&aux);
    */
#endif
  }
  DestroyRegionCloud3(&region);
}



bia::Scene8::Scene8 *SegmObjByMultipleClouds(Pyramid3 *pyr,
					     ArrayList *wpyr,
					     ArrayList *gpyr,
					     MRI_Info info,
					     char *basename,
					     char *bestname,
					     bia::Voxel pos_obj[]){
  bia::Scene8::Scene8 *label=NULL,*best_label=NULL;
  real score,best_scr=REAL_MIN;
  FileList *Gs=NULL;
  char filename[512];
  char groupname[512];
#ifndef _CLOUDSBRAIN_STANDALONE_
  char msg[512];
#endif
  bia::Voxel pos[4];
  int g;

  sprintf(filename,"%s/data/models/groups.txt",bia_dir);
  Gs = ReadFileList(filename);

  for(g=0; g<Gs->n; g++){
    sprintf(groupname,"%s_group%03d",basename,g+1);

#ifndef _CLOUDSBRAIN_STANDALONE_
    sprintf(msg,"Please wait, running Cloud System #%d...",g+1);
    APP->Busy(msg);
    APP->StatusMessage(msg);
#endif

    label = SegmObjBySingleCloud(pyr, wpyr, gpyr, info,
				 groupname, &score, pos);

#ifdef APPDEBUG
    sprintf(filename,"%s/data/debug/segm_triobj_%02d.scn",bia_dir,g+1);
    bia::Scene8::Write(label, filename);
#endif

    if(score>best_scr){
      best_scr = score;
      memcpy(pos_obj, pos, sizeof(bia::Voxel)*4);
      strcpy(bestname, groupname);
      if(best_label!=NULL) bia::Scene8::Destroy(&best_label);
      best_label = label;
    }
    else bia::Scene8::Destroy(&label);
  }
  DestroyFileList(&Gs);
  return best_label;
}


/*
bia::Scene8::Scene8 *SegmObjBySingleCloud(Pyramid3 *pyr,
					  ArrayList *wpyr,
					  ArrayList *gpyr,
					  MRI_Info info,
					  char *basename, 
					  real *score,
					  bia::Voxel pos_obj[]){
#ifdef APPTIME
  timer tic;
#endif
  bia::AdjSeedmap3::AdjSeedmap3 *asmap[NLAYERS];
  BorderCloud3 *bcloud[NLAYERS];
  IFTSC3AuxiliaryData *aux;
  Scene *scn,*tmp;
  bia::Scene8::Scene8 *label,*blabel,*flabel;
  bia::Scene16::Scene16 *arcw,*wobj;
  ScnGradient *grad;
  real cutmax,cut,fcut;
#ifdef APPDEBUG
  real *cuts;
  char filename[512];
  bia::Scene16::Scene16 *farcw=NULL;
#endif
  bia::AdjRel3::AdjRel3 *Adjs[NLAYERS],*Grid1,*Grid2,*A;
  int o,i,r,p,q,n,nobjs,lb;
  bia::Voxel u,v,dv,pos;
  bool fit;
  FIFOQ *Q=NULL;
  bia::BMap::BMap *bmap=NULL;
#ifndef _CLOUDSBRAIN_STANDALONE_
  int nrun=0,max_progr;
  wxProgressDialog *progress;
#endif

  ComputeCloudPyramid3(asmap, bcloud, basename, pyr);
  nobjs = asmap[0]->nobjs;

  scn  = GetPyramid3LayerRef(pyr, NLAYERS-1);
  Grid1 = bia::AdjRel3::SphericalGrid_mm(scn->dx, 
					 scn->dy, 
					 scn->dz, 
					 16.0, 8.0);
  Grid2 = bia::AdjRel3::SphericalGrid_mm(scn->dx,
					 scn->dy,
					 scn->dz,
					 4.0, 4.0);
  Adjs[NLAYERS-1] = bia::AdjRel3::Spheric(1.5);
  bia::AdjRel3::Scale(Adjs[NLAYERS-1], 
		      2./scn->dx,
		      2./scn->dy,
		      2./scn->dz);
  scn  = GetPyramid3LayerRef(pyr, 0);
  Adjs[0] = bia::AdjRel3::Spheric(0.5); //1.0
  bia::AdjRel3::Scale(Adjs[0], 
		      1./scn->dx,
		      1./scn->dy,
		      1./scn->dz);
#ifndef _CLOUDSBRAIN_STANDALONE_
  if(APP->Data.aligned){
    bia::AdjRel3::ClipX(Grid1,           0, 0);
    bia::AdjRel3::ClipX(Grid2,           0, 0);
    bia::AdjRel3::ClipX(Adjs[NLAYERS-1], 0, 0);
    bia::AdjRel3::ClipX(Adjs[0],         0, 0);
  }
#endif

  //Imagem de baixa resolucao.
  scn  = GetPyramid3LayerRef(pyr, NLAYERS-1);
  wobj = ((bia::Scene16::Scene16 *)
	  GetArrayListElement(wpyr, NLAYERS-1));
  arcw = bia::Scene16::Clone(wobj);
  grad = (ScnGradient *)GetArrayListElement(gpyr, NLAYERS-1);
  ComputeScnGradientMagnitude(grad);
  aux = CreateIFTSC3AuxiliaryData(arcw);

  n = scn->n;
#ifdef APPDEBUG
  cuts  = AllocRealArray(n);
#endif
  label = bia::Scene8::Create(scn->xsize,
			      scn->ysize,
			      scn->zsize);
  //----------------------------------------------
#ifndef _CLOUDSBRAIN_STANDALONE_
  max_progr = ( Grid1->n   + Grid2->n  +
	       (Adjs[1]->n + Adjs[0]->n)*nobjs);
  //printf("max_progr: %d\n",max_progr);

  progress = new wxProgressDialog(_T("Progress..."), _T(""),
				  max_progr,
				  APP->Window,
				  wxPD_AUTO_HIDE | wxPD_APP_MODAL);
  wxTheApp->Yield();
#endif
#ifdef APPTIME
  StartTimer(&tic);
#endif
  //----------------------------------------------

  Q = FIFOQNew(n/2);
  tmp = GetPyramid3LayerRef(pyr, 0);
  pos.c.x = ROUND((info.COG.x - tmp->xsize/2)/2.) + scn->xsize/2;
  pos.c.y = ROUND((info.COG.y - tmp->ysize/2)/2.) + scn->ysize/2;
  pos.c.z = ROUND((info.COG.z - tmp->zsize/2)/2.) + scn->zsize/2;
  p = VoxelAddress(scn,pos.c.x,pos.c.y,pos.c.z);

  FIFOQPush(Q, p);
  bmap = bia::BMap::Create(n);
  cutmax = REAL_MIN;

  A = Grid1;
  while(!FIFOQEmpty(Q)){
    p = FIFOQPop(Q);
    u.c.x = VoxelX(scn, p);
    u.c.y = VoxelY(scn, p);
    u.c.z = VoxelZ(scn, p);
    for(i=0; i<A->n; i++){
#ifndef _CLOUDSBRAIN_STANDALONE_
      progress->Update(nrun, _T(""), NULL);
      wxTheApp->Yield();
      nrun++;
#endif

      v.v = u.v + A->d[i].v;
      if(!ValidVoxel(scn,v.c.x,v.c.y,v.c.z)) continue;
      q = VoxelAddress(scn,v.c.x,v.c.y,v.c.z);
      if(bia::BMap::Get(bmap,q)) continue;

      fit = true;
      for(o=1; o<=nobjs; o++){
	dv.v = v.v + (asmap[NLAYERS-1]->disp)->d[o].v;
	fit = fit && bia::AdjRegion3::FitInside(asmap[NLAYERS-1]->obj_border[o], 
						dv, 
						scn->xsize, 
						scn->ysize, 
						scn->zsize, 0);
      }
      if(!fit){
	bia::BMap::Set1(bmap,q); 
	continue; 
      }
    
      cut = 0.0;
      for(o=1; o<=nobjs; o++){
	lb = o;
	bia::AdjSeedmap3::CloudArcWeight(arcw, wobj, grad, 
					 v, bcloud[NLAYERS-1],
					 asmap[NLAYERS-1], o, 0.08);
	cut += SeedmapIFTSC3(asmap[NLAYERS-1], o, lb, v, 
			     arcw, scn, label, info, aux);
      }
      cut /= (float)nobjs;

#ifdef APPDEBUG
      cuts[q] = cut;
#endif
    
      if(cut>cutmax){
	cutmax = cut;
	pos.v = v.v;
      }
      bia::BMap::Set1(bmap,q);
    }

    q = VoxelAddress(scn,pos.c.x,pos.c.y,pos.c.z);

    if(A==Grid1)  FIFOQPush(Q, q);
    if(A==Grid1)  A = Grid2;
  }

  //----------------------------------------------
#ifdef APPTIME
  printf("\n\tLowResTime: ");
  StopTimer(&tic);
#endif
  //----------------------------------------------

#ifdef APPDEBUG
  tmp = CreateScene(label->xsize, 
		    label->ysize, 
		    label->zsize);
  free(tmp->data);
  tmp->data = RealArray2IntArray(cuts, n, 0, 1000);
  sprintf(filename,"%s/data/debug/locate_%s.scn",
	  bia_dir,basename);
  WriteScene(tmp, filename);
  DestroyScene(&tmp);
  free(cuts);
#endif

#ifdef APPDEBUG
  for(o=1; o<=nobjs; o++){
    bia::Scene8::Fill(label, 0);
    bia::AdjSeedmap3::DrawObjBorder(label, asmap[NLAYERS-1], pos, o, 1);
    bia::AdjSeedmap3::DrawBkgBorder(label, asmap[NLAYERS-1], pos, o, 2);
    sprintf(filename,"%s/data/debug/seeds_%s_level%02d_%02d.scn",
	    bia_dir,basename,NLAYERS,o);
    bia::Scene8::Write(label, filename);
  }
#endif

  bia::BMap::Destroy(&bmap);
  FIFOQDestroy(Q);
  bia::Scene8::Destroy(&label);
  bia::Scene16::Destroy(&arcw);

  for(o=1; o<=nobjs; o++)
    pos_obj[o].v = pos.v;

  //Refinamento incluindo as imagens de mais alta resolucao (0<= r < NLAYERS-1).
  flabel = NULL;
  blabel = NULL;
  for(r=NLAYERS-1; r>=0; r--){

    A = Adjs[r];
    scn  = GetPyramid3LayerRef(pyr, r);
    wobj = ((bia::Scene16::Scene16 *)
	    GetArrayListElement(wpyr, r));
    arcw = bia::Scene16::Clone(wobj);
    grad = (ScnGradient *)GetArrayListElement(gpyr, r);
    ComputeScnGradientMagnitude(grad);
    if(aux==NULL) aux = CreateIFTSC3AuxiliaryData(arcw);

    label  = bia::Scene8::Create(scn->xsize,
				 scn->ysize,
				 scn->zsize);
    if(r==0){
      blabel = bia::Scene8::Create(label);
      flabel = bia::Scene8::Create(label);
    }
    //----------------------------------------------
#ifdef APPTIME
    StartTimer(&tic);
#endif
    //----------------------------------------------

    fcut = 0.0;
    for(o=1; o<=nobjs; o++){
      lb  = ROUND(pow(2.0,o-1));
      pos.v = pos_obj[o].v;

      if(r+1<=NLAYERS-1){
	tmp = GetPyramid3LayerRef(pyr, r+1);
	u.c.x = 2*(pos.c.x-tmp->xsize/2) + scn->xsize/2;
	u.c.y = 2*(pos.c.y-tmp->ysize/2) + scn->ysize/2;
	u.c.z = 2*(pos.c.z-tmp->zsize/2) + scn->zsize/2;
	pos.v = u.v;
      }

      u.v = pos.v;
      cutmax = REAL_MIN;
      for(i=0; i<A->n; i++){
#ifndef _CLOUDSBRAIN_STANDALONE_
	progress->Update(nrun, _T(""), NULL);
	wxTheApp->Yield();
	nrun++;
#endif

	v.v = u.v + A->d[i].v;
	if(!ValidVoxel(scn,v.c.x,v.c.y,v.c.z)) continue;

#ifdef APPDEBUG
	if(r==0)
	  bia::Scene16::Copy(arcw, wobj);
#endif
	bia::AdjSeedmap3::CloudArcWeight(arcw, wobj, grad,
					 v, bcloud[r],
					 asmap[r], o, 0.08);
	  
	cut = SeedmapIFTSC3(asmap[r], o, lb, v, arcw, 
			    scn, label, info, aux);
	
	if(cut>cutmax){
	  if(r==0){
#ifdef APPDEBUG
	    if(farcw!=NULL) bia::Scene16::Destroy(&farcw);
	    farcw = bia::Scene16::Clone(arcw);
#endif
	    bia::AdjSeedmap3::CopyUncertainty(blabel, label,
					      asmap[r], v, o);
	  }
	  cutmax = cut;
	  pos.v = v.v;
	}
      }
      if(r==0){
	bia::AdjSeedmap3::AddUncertainty(flabel, blabel,
					 asmap[r], pos, o);
	bia::AdjSeedmap3::DrawObject(flabel, asmap[r], pos, o, lb);
      }
      pos_obj[o].v = pos.v;
      fcut += cutmax;

#ifdef APPDEBUG
      if(r==0){
	sprintf(filename,"%s/data/debug/farcw_%s_%02d.scn",
		bia_dir,basename,o);
	bia::Scene16::Write(farcw, filename);
	bia::Scene16::Destroy(&farcw);
      }
#endif
    } //nobjs.
    fcut /= (float)nobjs;

    //----------------------------------------------
#ifdef APPTIME
    if(r==1) printf("\n\tMediResTime: ");
    if(r==0) printf("\n\tHighResTime: ");
    StopTimer(&tic);
#endif
    //----------------------------------------------

#ifdef APPDEBUG
    for(o=1; o<=nobjs; o++){
      bia::Scene8::Fill(label, 0);
      bia::AdjSeedmap3::DrawObjBorder(label, asmap[r], pos_obj[o], o, 1);
      bia::AdjSeedmap3::DrawBkgBorder(label, asmap[r], pos_obj[o], o, 2);
      sprintf(filename,"%s/data/debug/seeds_%s_level%02d_%02d.scn",
	      bia_dir,basename,r+1,o);
      bia::Scene8::Write(label, filename);
    }
#endif
    bia::Scene8::Destroy(&label);
    bia::Scene16::Destroy(&arcw);
    DestroyIFTSC3AuxiliaryData(&aux);
  } //resolution.

#ifndef _CLOUDSBRAIN_STANDALONE_
  progress->Update(nrun, _T(""), NULL);
  wxTheApp->Yield();
#endif
  bia::Scene8::Destroy(&blabel);
  bia::AdjRel3::Destroy(&Grid1);
  bia::AdjRel3::Destroy(&Grid2);
  for(i=0; i<NLAYERS; i++){
    bia::AdjRel3::Destroy(&Adjs[i]);
    bia::AdjSeedmap3::Destroy(&asmap[i]);
    DestroyBorderCloud3(&bcloud[i]);
  }
  *score = (real)fcut;

  return flabel;
}
*/

typedef struct _bestlocation{
  bia::Voxel pos;
  float cutmax;
} BestLocation;

#ifndef _CLOUDSBRAIN_STANDALONE_
typedef struct _searchprogress{
  int nrun;
  wxProgressDialog *dialog;
} SearchProgress;
#endif

typedef struct _argcloudsearch{
  wxMutex  *mutex;
  MRI_Info *info;
  int p;
  bia::AdjRel3::AdjRel3 *A;
  Scene *scn;
  bia::Scene16::Scene16 *wobj;
  ScnGradient *grad;
  bia::AdjSeedmap3::AdjSeedmap3 *asmap;
  BorderCloud3 *bcloud;
  bia::BMap::BMap *bmap;
  BestLocation *best;
#ifndef _CLOUDSBRAIN_STANDALONE_
  SearchProgress *progress;
#endif
  //----thread-dependent data-----
  int begin;
  int end;
} ArgCloudSearch;



void *ThreadCloudSearch(void *arg){
  ArgCloudSearch *p_arg;
  IFTSC3AuxiliaryData *aux;
  bia::AdjRel3::AdjRel3 *A;
  Scene *scn;
  bia::Scene16::Scene16 *wobj,*arcw;
  bia::Scene8::Scene8 *label;
  ScnGradient *grad;
  bia::AdjSeedmap3::AdjSeedmap3 *asmap;
  BorderCloud3 *bcloud;
  bia::BMap::BMap *bmap;
  wxMutex  *mutex;
  float cut; 
  bia::Voxel u,v,dv;
  int i,p,q,o,lb;
  bool fit,b;

  p_arg  = (ArgCloudSearch *)arg;
  mutex  = p_arg->mutex;
  A      = p_arg->A;
  scn    = p_arg->scn;
  wobj   = p_arg->wobj;
  grad   = p_arg->grad;
  asmap  = p_arg->asmap;
  bcloud = p_arg->bcloud;
  bmap   = p_arg->bmap;
  p = p_arg->p;

  arcw = bia::Scene16::Clone(wobj);
  aux = CreateIFTSC3AuxiliaryData(arcw);
  label = bia::Scene8::Create(scn->xsize,
			      scn->ysize,
			      scn->zsize);

  u.c.x = VoxelX(scn, p);
  u.c.y = VoxelY(scn, p);
  u.c.z = VoxelZ(scn, p);
  for(i=p_arg->begin; i<=p_arg->end; i++){
#ifndef _CLOUDSBRAIN_STANDALONE_
    mutex->Lock();
    (p_arg->progress->dialog)->Update(p_arg->progress->nrun,
				      _T(""), NULL);
    wxTheApp->Yield();
    (p_arg->progress->nrun)++;
    mutex->Unlock();
#endif

    v.v = u.v + A->d[i].v;
    if(!ValidVoxel(scn,v.c.x,v.c.y,v.c.z)) continue;
    q = VoxelAddress(scn,v.c.x,v.c.y,v.c.z);

    mutex->Lock();
    b = bia::BMap::Get(bmap,q);
    mutex->Unlock();

    if(b) continue;

    fit = true;
    for(o=1; o<=asmap->nobjs; o++){
      dv.v = v.v + (asmap->disp)->d[o].v;
      fit = fit && bia::AdjRegion3::FitInside(asmap->obj_border[o], 
					      dv, 
					      scn->xsize, 
					      scn->ysize, 
					      scn->zsize, 0);
    }
    if(!fit){
      mutex->Lock();
      bia::BMap::Set1(bmap,q); 
      mutex->Unlock();
      continue; 
    }
    
    cut = 0.0;
    for(o=1; o<=asmap->nobjs; o++){
      lb = o;
      bia::AdjSeedmap3::CloudArcWeight(arcw, wobj, grad, 
				       v, bcloud,
				       asmap, o, 0.08);
      cut += SeedmapIFTSC3(asmap, o, lb, v, 
			   arcw, scn, label, 
			   *(p_arg->info), aux);
    }
    cut /= (float)asmap->nobjs;

    mutex->Lock();
    if(cut>(p_arg->best->cutmax)){
      (p_arg->best)->cutmax = cut;
      (p_arg->best->pos).v = v.v;
    }
    bia::BMap::Set1(bmap,q);
    mutex->Unlock();
  }

  bia::Scene8::Destroy(&label);
  bia::Scene16::Destroy(&arcw);
  DestroyIFTSC3AuxiliaryData(&aux);

  return NULL;
}


void CloudSearch(ArgCloudSearch *S){
  ArgCloudSearch args[8];
  int nprocs,i;
  int first,last,nelems,de;
  pthread_t thread_id[8];
  int iret[8];

  nprocs = GetNumberOfProcessors();
  if(nprocs>=8) nprocs = 8;

  first  = 0;
  last   = (S->A->n)-1;
  nelems = last-first+1;
  de     = nelems/nprocs;
  S->begin = NIL;
  S->end   = first-1;

  for(i=0; i<nprocs; i++){
    args[i] = *S;
	
    args[i].begin = S->end+1;
    if(i<nprocs-1) args[i].end = args[i].begin+(de-1);
    else           args[i].end = last;

    //Create independent threads each of which will execute function
    iret[i] = pthread_create(&thread_id[i], NULL, 
			     ThreadCloudSearch,
			     (void*)&args[i]);
    S->begin = args[i].begin;
    S->end   = args[i].end;
  }
  
  //Wait till threads are complete before main continues.
  for(i=0; i<nprocs; i++)
    pthread_join(thread_id[i], NULL);
}



bia::Scene8::Scene8 *SegmObjBySingleCloud(Pyramid3 *pyr,
					  ArrayList *wpyr,
					  ArrayList *gpyr,
					  MRI_Info info,
					  char *basename, 
					  real *score,
					  bia::Voxel pos_obj[]){
#ifdef APPTIME
  timer tic;
#endif
  bia::AdjSeedmap3::AdjSeedmap3 *asmap[NLAYERS];
  BorderCloud3 *bcloud[NLAYERS];
  IFTSC3AuxiliaryData *aux=NULL;
  Scene *scn,*tmp;
  bia::Scene8::Scene8 *label,*blabel,*flabel;
  bia::Scene16::Scene16 *arcw,*wobj;
  ScnGradient *grad;
  real cutmax,cut,fcut;
#ifdef APPDEBUG
  char filename[512];
  bia::Scene16::Scene16 *farcw=NULL;
#endif
  bia::AdjRel3::AdjRel3 *Adjs[NLAYERS],*Grid1,*Grid2,*A;
  int o,i,r,n,nobjs,lb;
  bia::Voxel u,v,pos;
#ifndef _CLOUDSBRAIN_STANDALONE_
  int max_progr;
  wxProgressDialog *progress;
#endif
  wxMutex mutex;

  ComputeCloudPyramid3(asmap, bcloud, basename, pyr);
  nobjs = asmap[0]->nobjs;

  scn  = GetPyramid3LayerRef(pyr, NLAYERS-1);
  Grid1 = bia::AdjRel3::SphericalGrid_mm(scn->dx, 
					 scn->dy, 
					 scn->dz, 
					 16.0, 8.0);
  Grid2 = bia::AdjRel3::SphericalGrid_mm(scn->dx,
					 scn->dy,
					 scn->dz,
					 4.0, 4.0);
  Adjs[NLAYERS-1] = bia::AdjRel3::Spheric(1.5);
  bia::AdjRel3::Scale(Adjs[NLAYERS-1], 
		      2./scn->dx,
		      2./scn->dy,
		      2./scn->dz);
  scn  = GetPyramid3LayerRef(pyr, 0);
  Adjs[0] = bia::AdjRel3::Spheric(0.5); //1.0
  bia::AdjRel3::Scale(Adjs[0], 
		      1./scn->dx,
		      1./scn->dy,
		      1./scn->dz);
#ifndef _CLOUDSBRAIN_STANDALONE_
  if(APP->Data.aligned){
    bia::AdjRel3::ClipX(Grid1,           0, 0);
    bia::AdjRel3::ClipX(Grid2,           0, 0);
    bia::AdjRel3::ClipX(Adjs[NLAYERS-1], 0, 0);
    bia::AdjRel3::ClipX(Adjs[0],         0, 0);
  }
#endif

  //Imagem de baixa resolucao.
  scn  = GetPyramid3LayerRef(pyr, NLAYERS-1);
  wobj = ((bia::Scene16::Scene16 *)
	  GetArrayListElement(wpyr, NLAYERS-1));
  grad = (ScnGradient *)GetArrayListElement(gpyr, NLAYERS-1);
  ComputeScnGradientMagnitude(grad);

  n = scn->n;

  //----------------------------------------------
#ifndef _CLOUDSBRAIN_STANDALONE_
  max_progr = ( Grid1->n   + Grid2->n  +
	       (Adjs[1]->n + Adjs[0]->n)*nobjs);
  //printf("max_progr: %d\n",max_progr);

  progress = new wxProgressDialog(_T("Progress..."), _T(""),
				  max_progr,
				  APP->Window,
				  wxPD_AUTO_HIDE | wxPD_APP_MODAL);
  wxTheApp->Yield();
#endif
#ifdef APPTIME
  StartTimer(&tic);
#endif
  //----------------------------------------------

  ArgCloudSearch arg;
  MRI_Info layer_info = info;
  BestLocation location;

  tmp = GetPyramid3LayerRef(pyr, 0);
  arg.info = &layer_info;
  layer_info.COG.x = ROUND((info.COG.x - tmp->xsize/2)/2.) + scn->xsize/2;
  layer_info.COG.y = ROUND((info.COG.y - tmp->ysize/2)/2.) + scn->ysize/2;
  layer_info.COG.z = ROUND((info.COG.z - tmp->zsize/2)/2.) + scn->zsize/2;
  arg.best = &location;
  location.pos.c.x = layer_info.COG.x;
  location.pos.c.y = layer_info.COG.y;
  location.pos.c.z = layer_info.COG.z;
  location.cutmax  = REAL_MIN;
  arg.p = bia::Scene16::GetVoxelAddress(wobj, location.pos);
  arg.bmap = bia::BMap::Create(n);
  arg.mutex  = &mutex;
  arg.scn    = scn;
  arg.wobj   = wobj;
  arg.grad   = grad;
  arg.bcloud = bcloud[NLAYERS-1];
  arg.asmap  = asmap[NLAYERS-1];
  for(o=1; o<=nobjs; o++){
    bia::AdjRegion3::Optimize(arg.asmap->uncertainty[o], wobj);
    bia::AdjRegion3::Optimize(arg.asmap->bkg_border[o], wobj);
    bia::AdjRegion3::Optimize(arg.asmap->obj_border[o], wobj);
  }
#ifndef _CLOUDSBRAIN_STANDALONE_
  SearchProgress sprogr;
  arg.progress = &sprogr;
  arg.progress->nrun = 0;
  arg.progress->dialog = progress;
#endif

  arg.A = Grid1;
  CloudSearch(&arg);

  arg.A = Grid2;
  arg.p = bia::Scene16::GetVoxelAddress(wobj, location.pos);
  CloudSearch(&arg);
  pos.v = location.pos.v;

  bia::BMap::Destroy(&arg.bmap);

  //----------------------------------------------
#ifdef APPTIME
  printf("\n\tLowResTime: ");
  StopTimer(&tic);
#endif
  //----------------------------------------------

#ifdef APPDEBUG
  label = bia::Scene8::Create(scn->xsize,
			      scn->ysize,
			      scn->zsize);
  for(o=1; o<=nobjs; o++){
    bia::Scene8::Fill(label, 0);
    bia::AdjSeedmap3::DrawObjBorder(label, asmap[NLAYERS-1], pos, o, 1);
    bia::AdjSeedmap3::DrawBkgBorder(label, asmap[NLAYERS-1], pos, o, 2);
    sprintf(filename,"%s/data/debug/seeds_%s_level%02d_%02d.scn",
	    bia_dir,basename,NLAYERS,o);
    bia::Scene8::Write(label, filename);
  }
  bia::Scene8::Destroy(&label);
#endif

  aux = CreateIFTSC3AuxiliaryData(wobj);

  for(o=1; o<=nobjs; o++)
    pos_obj[o].v = pos.v;

  //Refinamento incluindo as imagens de mais alta resolucao (0<= r < NLAYERS-1).
  flabel = NULL;
  blabel = NULL;
  for(r=NLAYERS-1; r>=0; r--){

    A = Adjs[r];
    scn  = GetPyramid3LayerRef(pyr, r);
    wobj = ((bia::Scene16::Scene16 *)
	    GetArrayListElement(wpyr, r));
    arcw = bia::Scene16::Clone(wobj);
    grad = (ScnGradient *)GetArrayListElement(gpyr, r);
    ComputeScnGradientMagnitude(grad);
    if(aux==NULL) aux = CreateIFTSC3AuxiliaryData(arcw);

    label  = bia::Scene8::Create(scn->xsize,
				 scn->ysize,
				 scn->zsize);
    if(r==0){
      blabel = bia::Scene8::Create(label);
      flabel = bia::Scene8::Create(label);
    }
    //----------------------------------------------
#ifdef APPTIME
    StartTimer(&tic);
#endif
    //----------------------------------------------

    fcut = 0.0;
    for(o=1; o<=nobjs; o++){
      lb  = ROUND(pow(2.0,o-1));
      pos.v = pos_obj[o].v;

      if(r+1<=NLAYERS-1){
	tmp = GetPyramid3LayerRef(pyr, r+1);
	u.c.x = 2*(pos.c.x-tmp->xsize/2) + scn->xsize/2;
	u.c.y = 2*(pos.c.y-tmp->ysize/2) + scn->ysize/2;
	u.c.z = 2*(pos.c.z-tmp->zsize/2) + scn->zsize/2;
	pos.v = u.v;
      }

      u.v = pos.v;
      cutmax = REAL_MIN;
      for(i=0; i<A->n; i++){
#ifndef _CLOUDSBRAIN_STANDALONE_
	progress->Update(sprogr.nrun, _T(""), NULL);
	wxTheApp->Yield();
	sprogr.nrun++;
#endif

	v.v = u.v + A->d[i].v;
	if(!ValidVoxel(scn,v.c.x,v.c.y,v.c.z)) continue;

#ifdef APPDEBUG
	if(r==0)
	  bia::Scene16::Copy(arcw, wobj);
#endif
	bia::AdjSeedmap3::CloudArcWeight(arcw, wobj, grad,
					 v, bcloud[r],
					 asmap[r], o, 0.08);
	  
	cut = SeedmapIFTSC3(asmap[r], o, lb, v, arcw, 
			    scn, label, info, aux);
	
	if(cut>cutmax){
	  if(r==0){
#ifdef APPDEBUG
	    if(farcw!=NULL) bia::Scene16::Destroy(&farcw);
	    farcw = bia::Scene16::Clone(arcw);
#endif
	    bia::AdjSeedmap3::CopyUncertainty(blabel, label,
					      asmap[r], v, o);
	  }
	  cutmax = cut;
	  pos.v = v.v;
	}
      }
      if(r==0){
	bia::AdjSeedmap3::AddUncertainty(flabel, blabel,
					 asmap[r], pos, o);
	bia::AdjSeedmap3::DrawObject(flabel, asmap[r], pos, o, lb);
      }
      pos_obj[o].v = pos.v;
      fcut += cutmax;

#ifdef APPDEBUG
      if(r==0){
	sprintf(filename,"%s/data/debug/farcw_%s_%02d.scn",
		bia_dir,basename,o);
	bia::Scene16::Write(farcw, filename);
	bia::Scene16::Destroy(&farcw);
      }
#endif
    } //nobjs.
    fcut /= (float)nobjs;

    //----------------------------------------------
#ifdef APPTIME
    if(r==1) printf("\n\tMediResTime: ");
    if(r==0) printf("\n\tHighResTime: ");
    StopTimer(&tic);
#endif
    //----------------------------------------------

#ifdef APPDEBUG
    for(o=1; o<=nobjs; o++){
      bia::Scene8::Fill(label, 0);
      bia::AdjSeedmap3::DrawObjBorder(label, asmap[r], pos_obj[o], o, 1);
      bia::AdjSeedmap3::DrawBkgBorder(label, asmap[r], pos_obj[o], o, 2);
      sprintf(filename,"%s/data/debug/seeds_%s_level%02d_%02d.scn",
	      bia_dir,basename,r+1,o);
      bia::Scene8::Write(label, filename);
    }
#endif
    bia::Scene8::Destroy(&label);
    bia::Scene16::Destroy(&arcw);
    DestroyIFTSC3AuxiliaryData(&aux);
  } //resolution.

#ifndef _CLOUDSBRAIN_STANDALONE_
  progress->Update(sprogr.nrun, _T(""), NULL);
  wxTheApp->Yield();
#endif
  bia::Scene8::Destroy(&blabel);
  bia::AdjRel3::Destroy(&Grid1);
  bia::AdjRel3::Destroy(&Grid2);
  for(i=0; i<NLAYERS; i++){
    bia::AdjRel3::Destroy(&Adjs[i]);
    bia::AdjSeedmap3::Destroy(&asmap[i]);
    DestroyBorderCloud3(&bcloud[i]);
  }
  *score = (real)fcut;

  return flabel;
}


Scene *CopySceneWithInitialSize(Scene *scn, 
				int InitialSize[], 
				Voxel MBBmin){
  Scene *cpy=NULL;
  cpy = CreateScene(InitialSize[0],
		    InitialSize[1],
		    InitialSize[2]);
  PasteSubScene(cpy, scn, MBBmin);
  return cpy;
}


void WriteSceneWithInitialSize(Scene *scn, char *filename,
			       int InitialSize[], Voxel MBBmin){
  Scene *tmp=NULL;
  tmp = CopySceneWithInitialSize(scn, InitialSize, MBBmin);
  WriteScene(tmp, filename);
  DestroyScene(&tmp);
}


#ifdef _CLOUDSBRAIN_STANDALONE_
int main(int argc, char **argv){
  char basename[512];
  Scene *scn=NULL,*label=NULL;
  //------- check number of parameters ----------
  if(argc < 3){
    fprintf(stdout,"usage: cloudsbrain_bin <input_scene> <output_scene> [-1|-2]\n");
    fprintf(stdout,"scene:..... Input MR image.\n");
    fprintf(stdout,"-1..... Iteractive bias correction with one iteration.\n");
    fprintf(stdout,"-2..... Iteractive bias correction with two iteration.\n");
    exit(0);
  }
  //------- read and check parameters -----------
  scn = ReadVolume(argv[1]);

  strcpy(basename, "triobj");  //"brain");
  label = CloudsBrainSegmentation(scn, basename);
  if( argc == 3 ) {
    int p;
    Scene *mask = NULL;
    Scene *corr = NULL;
    for( p = 0; p < label->n; p++ ) {
      if( label->data[ p ] != 0 )
	label->data[ p ] = 1;
    }
    mask = DilateBinScene3( label, 3.0 );
    corr = InhomogeneityCorrection( scn, mask, 1.3/*pf*/, 15.3/*radius*/, 1.6/*rgrowth*/, 2.0/*compression*/, 
				    0/*T1_PROTOCOL*/, GetNumberOfProcessors(), false/*verbose*/ );
    for( p = 0; p < label->n; p++ ) {
      if( mask->data[ p ] == 0 )
	corr->data[ p ] = scn->data[ p ];
    }
    //WriteScene( corr, ( char * ) "corr.scn.bz2" );
    DestroyScene( &label );
    label = CloudsBrainSegmentation( corr, basename );
    DestroyScene( &mask );
    DestroyScene( &corr );
    if( argv[ 2 ][ 1 ] == '2' ) {
      for( p = 0; p < label->n; p++ ) {
	if( label->data[ p ] != 0 )
	  label->data[ p ] = 1;
      }
      mask = DilateBinScene3( label, 2.5 );
      corr = InhomogeneityCorrection( scn, mask, 1.3/*pf*/, 15.3/*radius*/, 1.6/*rgrowth*/, 2.0/*compression*/, 
				      0/*T1_PROTOCOL*/, GetNumberOfProcessors(), false/*verbose*/ );
      for( p = 0; p < label->n; p++ ) {
	if( mask->data[ p ] == 0 )
	  corr->data[ p ] = scn->data[ p ];
      }
      //WriteScene( corr, (char * ) "corr2.scn.bz2" );
      DestroyScene( &label );
      label = CloudsBrainSegmentation( corr, basename );
      DestroyScene( &mask );
      DestroyScene( &corr );
    }
  }
  CopySceneHeader( scn, label );
  WriteScene( label, argv[ 2 ] );

  DestroyScene(&label);
  DestroyScene(&scn);

  return 0;
}
#endif




//To smooth the inferior part of the structures.
void SmoothInferiorPart(Scene *label, int l1, int l2){
  BoxSelection3 box;
  Scene *fsub=NULL,*sub=NULL,*mask=NULL;
  Voxel COG;
  int l;

  mask = Threshold3(label, l1, l2);
  ComputeMaskCentroid3(mask, &COG);
  box.v1.x = 1; box.v2.x = label->xsize-2;
  box.v1.y = 1; box.v2.y = label->ysize-2;
  box.v1.z = 1; box.v2.z = COG.z;
  sub  = CopySubScene(mask, box.v1, box.v2);
  fsub = OpenBinScn(sub, 2.0);
  CopySceneFrame(fsub, sub, 1);
  PasteSubScene(mask, fsub, box.v1);
  DestroyScene(&sub);
  DestroyScene(&fsub);
  for(l=l1; l<=l2; l++)
    ClearLabelOutsideMask3(label, l, mask);
  DestroyScene(&mask);
}


void FixLabelRepresentation(Scene *label, int nobjs){
  int p,q,lb,o,i;
  int *vlb = AllocIntArray(nobjs+1);
  FIFOQ *Q=NULL;
  bia::Voxel u,v;
  bia::AdjRel3::AdjRel3 *A; 

  A = bia::AdjRel3::Spheric(1.0);
  vlb[0] = 0;
  for(o=1; o<=nobjs; o++)
    vlb[o] = ROUND(pow(2.0,o-1));

  for(p=0; p<label->n; p++){
    lb = label->data[p];
    label->data[p] = NIL;
    for(o=0; o<=nobjs; o++){
      if(lb==vlb[o]){ 
	label->data[p] = o;
	break; 
      }
    }
  }
  free(vlb);

  Q = FIFOQNew(label->n);

  for(p=0; p<label->n; p++){
    if(label->data[p]==NIL ||
       label->data[p]==0) continue;

    u.c.x = VoxelX(label, p);
    u.c.y = VoxelY(label, p);
    u.c.z = VoxelZ(label, p);
    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;
      if(ValidVoxel(label,v.c.x,v.c.y,v.c.z)){
	q = VoxelAddress(label,v.c.x,v.c.y,v.c.z);
	if(label->data[q]==NIL){
	  FIFOQPush(Q, p);
	  break;
	}
      }
    }
  }

  while(!FIFOQEmpty(Q)){
    p = FIFOQPop(Q);
    u.c.x = VoxelX(label, p);
    u.c.y = VoxelY(label, p);
    u.c.z = VoxelZ(label, p);

    for(i=1; i<A->n; i++){
      v.v = u.v + A->d[i].v;
      if(ValidVoxel(label,v.c.x,v.c.y,v.c.z)){
	q = VoxelAddress(label,v.c.x,v.c.y,v.c.z);
	if(label->data[q]==NIL){
	  label->data[q] = label->data[p];
	  FIFOQPush(Q, q);
	}
      }
    }
  }
  bia::AdjRel3::Destroy(&A);
  FIFOQDestroy(Q);
}


Scene *CloudsBrainSegmentation(Scene *orig, char *basename){
#ifdef APPTIME
  timer tic;
#endif
  Pyramid3 *pyr;
  ArrayList *wpyr;
  ArrayList *gpyr;
  bia::Scene8::Scene8 *label8;
  Scene *scn,*mask,*label,*tmp;
#ifndef _CLOUDSBRAIN_STANDALONE_
  bia::Scene16::Scene16 *scn16;
  ScnGradient *grad;
#endif
  char bestname[512];
#ifdef APPDEBUG
  char filename[512];
#endif
  bia::Voxel pos_obj[4];
  Voxel MBBmin,MBBmax;
  int InitialSize[3];
  MRI_Info info;
  int p;

#ifdef APPTIME
  StartTimer(&tic);
#endif

  bia_dir = getenv("BIA_DIR");
  if(bia_dir==NULL)
    Error((char*)"Environment variable BIA_DIR must be defined",
	  (char*)"CloudsBrainSegmentation");


  scn = orig;
  //---------------------------------------------
  InitialSize[0] = scn->xsize;
  InitialSize[1] = scn->ysize;
  InitialSize[2] = scn->zsize;
  ComputeMRISceneMBB(scn, &MBBmin, &MBBmax);
  tmp = CopySubScene(scn, MBBmin, MBBmax);
  scn = tmp;

#ifndef _CLOUDSBRAIN_STANDALONE_
  APP->Busy((char *)"Please wait, computing MRI information...");
  APP->StatusMessage((char *)"Please wait - Computing MRI Information...");
#endif

  info = EstimateMRI_Information(scn);
#ifndef _CLOUDSBRAIN_STANDALONE_
  if(APP->Data.aligned) (info.COG).x = scn->xsize/2;
#endif

#ifdef APPDEBUG
  printf("Tcsf: %d + %d\n",info.Tcsf,info.Ecsf);
  printf("Tgm:  %d\n",info.Tgm);
  printf("Twm:  %d\n",info.Twm);
  printf("Tup:  %d\n",info.Tup);
  printf("COG: (%d, %d, %d)\n",info.COG.x,info.COG.y,info.COG.z);
#endif

#ifndef _CLOUDSBRAIN_STANDALONE_
  APP->Busy((char *)"Please wait, computing pyramid...");
  APP->StatusMessage((char *)"Please wait - Computing Gaussian Pyramid...");
#endif
  pyr  = GaussianPyramid3(scn, NLAYERS);

#ifndef _CLOUDSBRAIN_STANDALONE_
  APP->Busy((char *)"Please wait, computing gradient...");
  APP->StatusMessage((char *)"Please wait - Computing Gradient...");
#endif
  gpyr = ComputeScnGradientPyramid3(pyr, info);
  wpyr = ComputeWeightPyramid3(pyr, gpyr, info);

  DestroyScene(&scn);

#ifdef APPDEBUG
  sprintf(filename,"%s/data/debug/scn",bia_dir);
  WritePyramid3Layers(pyr, filename);
  //sprintf(filename,"%s/data/debug/weight",bia_dir);
  //WritePyramid3Layers(wpyr, filename);
#endif

  //----------------------------------------------
#ifdef APPTIME
  printf("\nPreprocTime: ");
  StopTimer(&tic);

  StartTimer(&tic);
#endif
  //----------------------------------------------

  label8 = SegmObjByMultipleClouds(pyr, wpyr, gpyr, 
				   info, basename, 
				   bestname, pos_obj);
  label = CreateScene(label8->xsize,
		      label8->ysize,
		      label8->zsize);
  label->dx = label8->dx;
  label->dy = label8->dy;
  label->dz = label8->dz;
  for(p=0; p<label->n; p++)
    label->data[p] = (int)label8->data[p];
  bia::Scene8::Destroy(&label8);

  //----------------------------------------------
#ifdef APPTIME
  printf("\nMultipleCloudsTime: ");
  StopTimer(&tic);

  StartTimer(&tic);
#endif
  //----------------------------------------------

#ifdef APPDEBUG
  printf("best: %s\n",bestname);
#endif

#ifdef APPDEBUG
  sprintf(filename,"%s/data/segm_brain_before.scn",bia_dir);
  WriteSceneWithInitialSize(label, filename,
			    InitialSize, MBBmin);
#endif

  scn = GetPyramid3LayerRef(pyr,  0);

  //Post-processing (remove background):
#ifndef _CLOUDSBRAIN_STANDALONE_
  APP->Busy((char *)"Please wait, running post-processing...");
  APP->StatusMessage((char *)"Please wait - running post-processing...");
#endif

  FixLabelRepresentation(label, 3);

  mask = Threshold3(label, 1, 1);
  ClearBinBelowThreshold(scn, mask, info.Tcsf+info.Ecsf/2);
  ClearLabelOutsideMask3(label, 1, mask);
  DestroyScene(&mask);

  mask = Threshold3(label, 2, 3);
  ClearPeripheralBinBelowThreshold(scn, mask, 1.0,
				   info.Tcsf+info.Ecsf/2);
  ClearLabelOutsideMask3(label, 2, mask);
  ClearLabelOutsideMask3(label, 3, mask);
  DestroyScene(&mask);

  tmp = MRI_FillObjInterfaces(label, scn, 2.0, info);
  DestroyScene(&label);
  label = tmp;

  //tmp = SmoothObjInterfaces3(label, 2.0);
  //DestroyScene(&label);
  //label = tmp;

  SmoothInferiorPart(label, 2, 3); //Telencephalon.
  SmoothInferiorPart(label, 1, 1); //Cerebellum.
  ModeFilterLabel3(label, 2.0);

  mask = Threshold3(label, 2, 2);
  SelectLargestComp(mask);
  ClearLabelOutsideMask3(label, 2, mask);
  DestroyScene(&mask);

  mask = Threshold3(label, 3, 3);
  SelectLargestComp(mask);
  ClearLabelOutsideMask3(label, 3, mask);
  DestroyScene(&mask);

  //----------------------------------------------
#ifdef APPTIME
  printf("\nPostprocTime: ");
  StopTimer(&tic);
#endif
  //----------------------------------------------

#ifndef _CLOUDSBRAIN_STANDALONE_
  scn16 = (bia::Scene16::Scene16 *)GetArrayListElement(wpyr, 0);
  tmp = CreateScene(scn16->xsize, scn16->ysize, scn16->zsize);
  for(p=0; p<scn16->n; p++)
    tmp->data[p] = (int)scn16->data[p];

  if(APP->Data.arcw!=NULL)
    DestroyScene(&(APP->Data.arcw));
  APP->Data.arcw = CopySceneWithInitialSize(tmp, InitialSize, MBBmin);
  MaximumValue3(APP->Data.arcw);

  DestroyScene(&tmp);

  grad = (ScnGradient *)GetArrayListElement(gpyr, 0);
  DestroyScnGradient(&(APP->Data.grad));
  APP->Data.grad = (ScnGradient *) calloc(1,sizeof(ScnGradient));
  if(APP->Data.grad == NULL) Error((char*) MSG1, (char*) "CloudsBrainSegmentation");
  (APP->Data.grad)->Gx  = CopySceneWithInitialSize(grad->Gx, InitialSize, MBBmin);
  (APP->Data.grad)->Gy  = CopySceneWithInitialSize(grad->Gy, InitialSize, MBBmin);
  (APP->Data.grad)->Gz  = CopySceneWithInitialSize(grad->Gz, InitialSize, MBBmin);
  (APP->Data.grad)->mag = CopySceneWithInitialSize(grad->mag, InitialSize, MBBmin);
  ScnGradientMaximumMag(APP->Data.grad);
#endif

  tmp = CopySceneWithInitialSize(label, InitialSize, MBBmin);

  DestroyScene(&label);
  DestroyPyramid3(&pyr);
  DestroyArrayList(&wpyr);
  DestroyArrayList(&gpyr);

  return(tmp);
}


