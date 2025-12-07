
#include "livehandler.h"

namespace Interactive{

  LiveHandler :: LiveHandler(char axis,
			     ModuleInteractive *mod){
    this->axis = axis;
    this->mod = mod;
    this->frontier = NULL;
    this->arcw  = NULL;
    this->pred  = NULL;
    this->cost  = NULL;
    this->label = NULL;
    this->A     = Circular(1.5);
    this->axis_val = NIL;
    this->Wmax     = NIL;
    this->closed = false;
    this->path_voxels = NULL;
    this->path_prevlb = NULL;
    this->ClearAnchorPoints();
  }

  LiveHandler :: ~LiveHandler(){
    if(frontier!=NULL) DestroyPQueue(&frontier);
    if(arcw!=NULL)  DestroyImage(&arcw);
    if(pred!=NULL)  DestroyImage(&pred);
    if(cost!=NULL)  DestroyImage(&cost);
    if(label!=NULL) DestroyImage(&label);
    if(A!=NULL)     DestroyAdjRel(&A);
    if(path_voxels!=NULL) free(path_voxels);
    if(path_prevlb!=NULL) free(path_prevlb);
  }

  void LiveHandler :: OnEnterWindow(){
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'x');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'y');
    APP->Set2DViewCursor(wxSTANDARD_CURSOR, 'z');
  }

  void LiveHandler :: OnLeaveWindow(){}

  void LiveHandler :: OnLeftClick(int p){
    int *boundary=NULL;
    Voxel u;
    Pixel a;
    int pxl;
    int q;

    a = GetCorrespondingPixel(p);
    u.x = VoxelX(APP->Data.orig, p);
    u.y = VoxelY(APP->Data.orig, p);
    u.z = VoxelZ(APP->Data.orig, p);

    //-------Axis X:--------------
    if(axis=='x'){
      if(axis_val!=u.x){
	if(arcw!=NULL) DestroyImage(&arcw);
	axis_val = u.x;
      }
      if(arcw==NULL){
	arcw = this->GetXSlice(APP->Data.arcw, u.x);
	Wmax = NIL;
      }
    }
    //-------Axis Y:--------------
    else if(axis=='y'){
      if(axis_val!=u.y){
	if(arcw!=NULL) DestroyImage(&arcw);
	axis_val = u.y;
      }
      if(arcw==NULL){
	arcw = this->GetYSlice(APP->Data.arcw, u.y);
	Wmax = NIL;
      }
    }
    //-------Axis Z:--------------
    else if(axis=='z'){
      if(axis_val!=u.z){
	if(arcw!=NULL) DestroyImage(&arcw);
	axis_val = u.z;
      }
      if(arcw==NULL){
	arcw = this->GetZSlice(APP->Data.arcw, u.z);
	Wmax = NIL;
      }
    }
    else return;

    //Live wire uses complemented weights.
    if(Wmax==NIL){
      Wmax = MaximumValue(arcw);
      for(q=0; q<arcw->ncols*arcw->nrows; q++)
	arcw->val[q] = Wmax - arcw->val[q];

      if(pred!=NULL)  DestroyImage(&pred);
      if(cost!=NULL)  DestroyImage(&cost);
      if(label!=NULL) DestroyImage(&label);
      pred  = CreateImage(arcw->ncols, arcw->nrows);
      cost  = CreateImage(arcw->ncols, arcw->nrows);
      label = CreateImage(arcw->ncols, arcw->nrows);
      SetImage(pred,  NIL);
      SetImage(cost,  INT_MAX);
      SetImage(label, 0);
      this->ClearAnchorPoints();
    }

    pxl = a.x + a.y*arcw->ncols;
    boundary = LiveWire(pxl, true, false, false);

    if(boundary!=NULL) free(boundary);    
    APP->Refresh2DCanvas(axis);
  }


  void LiveHandler :: OnRightClick(int p){
    int *boundary=NULL;
    Voxel u;
    Pixel a;
    int pxl;

    if(arcw==NULL) return;
    u.x = VoxelX(APP->Data.orig, p);
    u.y = VoxelY(APP->Data.orig, p);
    u.z = VoxelZ(APP->Data.orig, p);

    if(axis=='x'){
      if(axis_val!=u.x) return;
    }
    else if(axis=='y'){
      if(axis_val!=u.y) return;
    }
    else if(axis=='z'){
      if(axis_val!=u.z) return;
    }
    else return;

    a = GetCorrespondingPixel(p);
    pxl = a.x + a.y*arcw->ncols;
    boundary = LiveWire(pxl, false, false, true);

    RefreshPath(boundary);

    if(boundary!=NULL)
      free(boundary);
  }

  void LiveHandler :: OnMiddleClick(int p){
    Voxel u;
    Pixel a;
    int *boundary=NULL,*cpath=NULL;
    int pxl,src;

    src = GetLastAnchorPoint();
    if(arcw==NULL) return;
    if(src==NIL)   return;

    u.x = VoxelX(APP->Data.orig, p);
    u.y = VoxelY(APP->Data.orig, p);
    u.z = VoxelZ(APP->Data.orig, p);

    if(axis=='x'){
      if(axis_val!=u.x) return;
    }
    else if(axis=='y'){
      if(axis_val!=u.y) return;
    }
    else if(axis=='z'){
      if(axis_val!=u.z) return;
    }
    else return;
    
    a = GetCorrespondingPixel(p);
    pxl = a.x + a.y*arcw->ncols;
    boundary = LiveWire(pxl, false, false, false);

    RefreshPath(NULL);

    cpath = ConfirmedPathArray(boundary, src);
    if(boundary!=NULL) free(boundary);
    if(cpath==NULL) return;
    this->ClearAnchorPoints();
    if(pred!=NULL)  SetImage(pred,  NIL);
    if(cost!=NULL)  SetImage(cost,  INT_MAX);
    if(label!=NULL) SetImage(label, 0);

    this->InsertPathSeeds(cpath);
    if(cpath!=NULL) free(cpath);

    APP->Refresh2DCanvas(axis);

    mod->Run();
  }


  void LiveHandler :: OnMouseWheel(int p, int rotation, int delta){
  }


  void LiveHandler :: OnLeftDrag(int p, int q){
  }

  void LiveHandler :: OnRightDrag(int p, int q){
  }

  void LiveHandler :: OnMouseMotion(int p){
    Voxel u;
    Pixel a;
    int *boundary=NULL;
    int pxl;

    if(arcw==NULL) return;
    u.x = VoxelX(APP->Data.orig, p);
    u.y = VoxelY(APP->Data.orig, p);
    u.z = VoxelZ(APP->Data.orig, p);

    if(axis=='x'){
      if(axis_val!=u.x) return;
    }
    else if(axis=='y'){
      if(axis_val!=u.y) return;
    }
    else if(axis=='z'){
      if(axis_val!=u.z) return;
    }
    else return;

    a = GetCorrespondingPixel(p);
    pxl = a.x + a.y*arcw->ncols;
    boundary = LiveWire(pxl, false, false, false);
    
    RefreshPath(boundary);

    if(boundary!=NULL)
      free(boundary); 

    APP->Window->SetStatusText(_T("Mouse Left: Add Border Marker, Mouse Center: Accept and Run, Mouse Right: Del Last Marker"));
  }


  int *LiveHandler :: LiveWire(int px, 
			       bool add_click, 
			       bool close_click, 
			       bool del_click){
    int *boundary=NULL;
    int src,dst,init;
    Image *flabel=NULL;
    int n,i,p;
    //wxMutexLocker lock(APP->mutex);
    //---------------------
    //AdjRel *A = Circular(1.0);
    //---------------------
    
    if(cost->val[px]==INT_MIN) return NULL;
    //if(APP->IsInternalSeed(px, 1)) return NULL;
    if(IsAnchorPoint(px)) return NULL;
    if(add_click && close_click) return NULL;
    if(del_click && add_click) return NULL;
    if(del_click && close_click) return NULL;
    if(closed && !del_click) return NULL;
   
    //src = APP->GetLastInternalSeed(1);
    src = this->GetLastAnchorPoint();
    //init = APP->GetFirstInternalSeed(1);
    init = this->GetFirstAnchorPoint();

    if(close_click==true || closed){
      if(frontier!=NULL) ResetPQueue(frontier);
      for(p=0; p<arcw->ncols*arcw->nrows; p++){
	if(cost->val[p]>INT_MIN)
	  cost->val[p] = INT_MAX;
      }
      cost->val[init] = INT_MAX;
      dst = init;
    }
    else
      dst = px;
    if(dst==NIL) return NULL;

    if(src!=NIL){
      
      boundary = path_by_iftLiveWire(cost, pred,
				     arcw, A, this->Wmax,
				     &frontier,
				     init, src, dst);
      dst = boundary[1];

      SetImage(label, 0);
      if(boundary!=NULL){
	n = boundary[0];
	for(i=1; i<=n; i++){
	  p = boundary[i];
	  label->val[p] = 1;
	}
      }
    }
    
    if(add_click){
      if(frontier!=NULL)
	DestroyPQueue(&frontier);
      frontier = NULL;
      SetImage(cost, INT_MAX);
      SetImage(label,      0);
      SetImage(pred,     NIL);
      
      if(boundary!=NULL){
	n = boundary[0];
	for(i=1; i<=n; i++){
	  p = boundary[i];
	  //----------------------------------
	  //if(i<n && i>1)
	  //  DrawAdjRel(cost, A, p, INT_MIN);
	  //----------------------------------
	  cost->val[p] = INT_MIN;
	  label->val[p] = 1;
	  if(i<n)
	    pred->val[p] = boundary[i+1];
	}
      }
      
      //if(APP->GetSeedId(1, dst)==0){
      // APP->Data.markerID++;
      // APP->AddSeed(1, dst, 1, APP->Data.markerID);
      //}
      if(!IsAnchorPoint(dst))
	this->AppendAnchorPoint(dst);
    }

    if(close_click){
      //Test closed boundary:
      if(pred->val[init]!=NIL){
	closed = true;
	flabel = CloseHoles(label);
	memcpy(label->val,flabel->val,
	       flabel->ncols*flabel->nrows*sizeof(int));
	DestroyImage(&flabel);
      }
      //view->Update();
    }

    if(del_click){
      if(frontier!=NULL)
	DestroyPQueue(&frontier);
      frontier = NULL;
      closed = false;

      if(src!=NIL){
	//APP->Data.markerID--;
	//APP->DelSeed(1, src);
	this->DelLastAnchorPoint();
      }
      //src = APP->GetLastInternalSeed(1);
      src = this->GetLastAnchorPoint();

      SetImage(cost, INT_MAX);
      SetImage(label,      0);
      SetImage(pred,     NIL);

      if(boundary!=NULL){
	n = boundary[0];
	for(i=1; i<=n; i++)
	  if(boundary[i]==src) break;
	
	for(; i<=n; i++){
	  p = boundary[i];
	  //----------------------------------
	  //if(i<n && boundary[i]!=src)
	  //  DrawAdjRel(cost, A, p, INT_MIN);
	  //----------------------------------
	  cost->val[p] = INT_MIN;
	  label->val[p] = 1;
	  if(i<n)
	    pred->val[p] = boundary[i+1];
	}
      }
    }
    
    //if(boundary!=NULL)
    //  free(boundary);
    return boundary;
    //----------------------------
    //DestroyAdjRel(&A);
    //----------------------------
    
    //view->Update();
  }


  Pixel LiveHandler :: GetCorrespondingPixel(int p_scn){
    Voxel u;
    Pixel a;
    u.x = VoxelX(APP->Data.orig, p_scn);
    u.y = VoxelY(APP->Data.orig, p_scn);
    u.z = VoxelZ(APP->Data.orig, p_scn);
    if(axis=='x'){
      a.x = u.z;
      a.y = u.y;
    }
    else if(axis=='y'){
      a.x = u.x;
      a.y = u.z;
    }
    else{ //axis=='z'
      a.x = u.x;
      a.y = u.y;
    }
    return a;
  }

  Voxel LiveHandler :: GetCorrespondingVoxel(int p_img){
    Voxel u;
    Pixel a;
    a.x = p_img%arcw->ncols;
    a.y = p_img/arcw->ncols;
    if(axis=='x'){
      u.x = axis_val;
      u.y = a.y;
      u.z = a.x;
    }
    else if(axis=='y'){
      u.x = a.x;
      u.y = axis_val;
      u.z = a.y;
    }
    else{ //axis=='z'
      u.x = a.x;
      u.y = a.y;
      u.z = axis_val;
    }
    return u;
  }



  Image *LiveHandler :: GetZSlice(Scene *scn, int z){
    Image *img=NULL;
    int n;
    int *data;
    
    img  = CreateImage(scn->xsize,scn->ysize);
    n    = img->ncols*img->nrows;
    data = scn->data + z*n;
    n    = n*sizeof(int);
    memcpy(img->val, data, n);
    
    return(img);
  }


  Image *LiveHandler :: GetXSlice(Scene *scn, int x){
    Image *img=NULL;
    int i,j;
    Voxel v;

    v.x = x;
    img = CreateImage(scn->zsize,scn->ysize);
    for(v.z=0;v.z<scn->zsize;v.z++)
      for(v.y=0;v.y<scn->ysize;v.y++) {
	i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	if(scn->data[i] != 0) {
	  j = img->tbrow[v.y] + v.z;
	  img->val[j] = scn->data[i];
	}
      }
    return(img);
  }


  Image *LiveHandler :: GetYSlice(Scene *scn, int y){
    Image *img=NULL;
    int i,j;
    Voxel v;

    v.y = y;
    img = CreateImage(scn->xsize,scn->zsize);
    for (v.z=0;v.z<scn->zsize;v.z++)
      for (v.x=0;v.x<scn->xsize;v.x++) {
	i = scn->tbz[v.z] + scn->tby[v.y] + v.x;
	if (scn->data[i] != 0) {
	  j = img->tbrow[v.z] + v.x;
	  img->val[j] = scn->data[i];
	}
      }
    return(img);
  }


  void LiveHandler :: RefreshPath(int *boundary){
    Voxel v;
    int i,n,q;

    APP->SetLabelColour(3, 0xff0000, 255);

    //Restore previous label.
    if(path_voxels!=NULL){
      n = path_voxels[0];
      for(i=1; i<=n; i++){
	q = path_voxels[i];
	(APP->Data.label)->data[q] = path_prevlb[i];
      }
    }
    if(path_voxels!=NULL) free(path_voxels);
    if(path_prevlb!=NULL) free(path_prevlb);
    path_voxels = NULL;
    path_prevlb = NULL;

    if(boundary!=NULL){
      n = boundary[0];
      path_voxels = AllocIntArray(n+1);
      path_prevlb = AllocIntArray(n+1);
      path_voxels[0] = n;
      path_prevlb[0] = n;

      for(i=1; i<=n; i++){
	v = GetCorrespondingVoxel(boundary[i]);
	q = VoxelAddress(APP->Data.orig,v.x,v.y,v.z);
	path_voxels[i] = q;
	path_prevlb[i] = (APP->Data.label)->data[q];
	(APP->Data.label)->data[q] = 3;
      }
    }
    APP->Refresh2DCanvas(axis);
  }


  void LiveHandler :: InsertPathSeeds(int *cpath){
    Image *tmp=NULL;
    Pixel a,b,c;
    Voxel u;
    int n,i,j,p,q,r,dx,dy;
    AdjRel *A8,*A4,*R,*L,*B,*A=NULL;
    FIFOQ *Q=NULL;
    int o,cod_obj,cod_bkg,Bsz;

    o = mod->obj_sel;
    if(o<0) return;
    //------------------------------------

    A8  = Circular(1.5);
    A4  = Circular(1.0);
    R   = RightSide(A8);
    L   = LeftSide(A8);
    B   = mod->GetBrush();
    Bsz = FrameSize(B);

    if(Bsz<=1){
      DestroyAdjRel(&B);
      B = Circular(1.5);
      Bsz = 1;
    }

    tmp = CreateImage(arcw->ncols, arcw->nrows);
    SetImage(tmp, INT_MAX);
    Q = FIFOQNew(tmp->ncols*tmp->nrows);

    n = cpath[0];
    for(i=1; i<=n; i++){
      p = cpath[i];
      a.x = p%tmp->ncols;
      a.y = p/tmp->ncols;
      tmp->val[p] = 2;
      if(i==1 || i==n) continue;

      if((i-1)>Bsz+1 && (n-i)>Bsz+1)
	A = B;
      else
	A = A8;

      for(j=1; j<A->n; j++){
	b.x = a.x + A->dx[j];
	b.y = a.y + A->dy[j];
	if(ValidPixel(tmp, b.x, b.y)){
	  q = b.x + b.y*tmp->ncols;
	  if(tmp->val[q]!=2)
	    tmp->val[q] = NIL;
	}
      }
    }

    for(i=2; i<n; i++){
      p = cpath[i];
      a.x = p%tmp->ncols;
      a.y = p/tmp->ncols;

      //Vector (dx,dy) from i-1 to i:
      q = cpath[i-1];
      b.x = q%tmp->ncols;
      b.y = q/tmp->ncols;
      dx = a.x - b.x;
      dy = a.y - b.y;
      
      for(j=1; j<A8->n; j++){
	if(A8->dx[j]==dx &&
	   A8->dy[j]==dy)
	  break;
      }
      if(j<R->n){
	c.x = b.x + R->dx[j];
	c.y = b.y + R->dy[j];
	if(ValidPixel(tmp, c.x, c.y)){
	  r = c.x + c.y*tmp->ncols;
	  if(tmp->val[r]==NIL){
	    tmp->val[r] = 1;
	    FIFOQPush(Q, r);
	  }
	  else if(tmp->val[r]==0){
	    //Collision detected
	    tmp->val[r] = 3;
	  }
	}
	c.x = b.x + L->dx[j];
	c.y = b.y + L->dy[j];
	if(ValidPixel(tmp, c.x, c.y)){
	  r = c.x + c.y*tmp->ncols;
	  if(tmp->val[r]==NIL){
	    tmp->val[r] = 0;
	    FIFOQPush(Q, r);
	  }
	  else if(tmp->val[r]==1){
	    //Collision detected
	    tmp->val[r] = 3;
	  }
	}
      }

      //Vector (dx,dy) from i to i+1:
      q = cpath[i+1];
      b.x = q%tmp->ncols;
      b.y = q/tmp->ncols;
      dx = b.x - a.x;
      dy = b.y - a.y;

      for(j=1; j<A8->n; j++){
	if(A8->dx[j]==dx &&
	   A8->dy[j]==dy)
	  break;
      }
      if(j<R->n){
	c.x = a.x + R->dx[j];
	c.y = a.y + R->dy[j];
	if(ValidPixel(tmp, c.x, c.y)){
	  r = c.x + c.y*tmp->ncols;
	  if(tmp->val[r]==NIL){
	    tmp->val[r] = 1;
	    FIFOQPush(Q, r);
	  }
	  else if(tmp->val[r]==0){
	    //Collision detected
	    tmp->val[r] = 3;
	  }
	}
	c.x = a.x + L->dx[j];
	c.y = a.y + L->dy[j];
	if(ValidPixel(tmp, c.x, c.y)){
	  r = c.x + c.y*tmp->ncols;
	  if(tmp->val[r]==NIL){
	    tmp->val[r] = 0;
	    FIFOQPush(Q, r);
	  }
	  else if(tmp->val[r]==1){
	    //Collision detected
	    tmp->val[r] = 3;
	  }
	}
      }
    }

    while(!FIFOQEmpty(Q)){
      p = FIFOQPop(Q);
      a.x = p%tmp->ncols;
      a.y = p/tmp->ncols;
      
      for(i=1; i<A4->n; i++){
	b.x = a.x + A4->dx[i];
	b.y = a.y + A4->dy[i];
	if(ValidPixel(tmp, b.x, b.y)){
	  q = b.x + b.y*tmp->ncols;
	  if(tmp->val[q]==NIL ||
	     tmp->val[q]==3){
	    tmp->val[q] = tmp->val[p];
	    FIFOQPush(Q, q);
	  }
	}
      }
    }

    //---------------------------------
    mod->markerID++;
    cod_obj = mod->GetCodeValue(mod->markerID, o+1);
    APP->SetLabelColour(cod_obj, mod->obj_color[o]);

    mod->markerID++;
    cod_bkg = mod->GetCodeValue(mod->markerID, 0);
    APP->SetLabelColour(cod_bkg, NIL);

    for(a.y=0; a.y<tmp->nrows; a.y++){
      for(a.x=0; a.x<tmp->ncols; a.x++){
	p = a.x + a.y*tmp->ncols;
	if(tmp->val[p]==1 ||
	   tmp->val[p]==2 ||
	   tmp->val[p]==0){
	  u = GetCorrespondingVoxel(p);
	  q = VoxelAddress(APP->Data.orig,u.x,u.y,u.z);
	  if(tmp->val[p]==1 ||
	     tmp->val[p]==2){
	    APP->AddSeed(q, o+1, mod->markerID-1);
	    (APP->Data.label)->data[q] = cod_obj;
	  }
	  else if(tmp->val[p]==0){
	    APP->AddSeed(q, 0, mod->markerID);
	    (APP->Data.label)->data[q] = cod_bkg;
	  }
	}
      }
    }

    FIFOQDestroy(Q);
    DestroyAdjRel(&A8);
    DestroyAdjRel(&A4);
    DestroyAdjRel(&R);
    DestroyAdjRel(&L);
    DestroyAdjRel(&B);
    DestroyImage(&tmp);
  }



  /*
  void LiveHandler :: InsertPathSeeds(int *cpath){
    Image *tmp=NULL;
    Pixel a,b,c;
    Voxel u;
    int n,i,j,p,q,r,dx,dy;
    AdjRel *A8 = Circular(1.5);
    AdjRel *A4 = Circular(1.0);
    AdjRel *R  = RightSide(A8);
    FIFOQ *Q=NULL;
    int o,cod_obj,cod_bkg;

    o = mod->obj_sel;
    if(o<0) return;
    //------------------------------------

    tmp = CreateImage(arcw->ncols, arcw->nrows);
    SetImage(tmp, NIL);
    Q = FIFOQNew(tmp->ncols*tmp->nrows);

    n = cpath[0];
    for(i=1; i<=n; i++){
      p = cpath[i];
      a.x = p%tmp->ncols;
      a.y = p/tmp->ncols;
      tmp->val[p] = 1;
      if(i==1 || i==n) continue;
      for(j=1; j<A8->n; j++){
	b.x = a.x + A8->dx[j];
	b.y = a.y + A8->dy[j];
	if(ValidPixel(tmp, b.x, b.y)){
	  q = b.x + b.y*tmp->ncols;
	  if(tmp->val[q]!=1)
	    tmp->val[q] = 0;
	}
      }
    }

    for(i=2; i<n; i++){
      p = cpath[i];
      a.x = p%tmp->ncols;
      a.y = p/tmp->ncols;

      //Vector (dx,dy) from i-1 to i:
      q = cpath[i-1];
      b.x = q%tmp->ncols;
      b.y = q/tmp->ncols;
      dx = a.x - b.x;
      dy = a.y - b.y;
      
      for(j=1; j<A8->n; j++){
	if(A8->dx[j]==dx &&
	   A8->dy[j]==dy)
	  break;
      }
      if(j<R->n){
	c.x = b.x + R->dx[j];
	c.y = b.y + R->dy[j];
	if(ValidPixel(tmp, c.x, c.y)){
	  r = c.x + c.y*tmp->ncols;
	  if(tmp->val[r]==0){
	    tmp->val[r] = 1;
	    FIFOQPush(Q, r);
	  }
	}
      }

      //Vector (dx,dy) from i to i+1:
      q = cpath[i+1];
      b.x = q%tmp->ncols;
      b.y = q/tmp->ncols;
      dx = b.x - a.x;
      dy = b.y - a.y;

      for(j=1; j<A8->n; j++){
	if(A8->dx[j]==dx &&
	   A8->dy[j]==dy)
	  break;
      }
      if(j<R->n){
	c.x = a.x + R->dx[j];
	c.y = a.y + R->dy[j];
	if(ValidPixel(tmp, c.x, c.y)){
	  r = c.x + c.y*tmp->ncols;
	  if(tmp->val[r]==0){
	    tmp->val[r] = 1;
	    FIFOQPush(Q, r);
	  }
	}
      }
    }

    while(!FIFOQEmpty(Q)){
      p = FIFOQPop(Q);
      a.x = p%tmp->ncols;
      a.y = p/tmp->ncols;
      
      for(i=1; i<A4->n; i++){
	b.x = a.x + A4->dx[i];
	b.y = a.y + A4->dy[i];
	if(ValidPixel(tmp, b.x, b.y)){
	  q = b.x + b.y*tmp->ncols;
	  if(tmp->val[q]==0){
	    tmp->val[q] = 1;
	    FIFOQPush(Q, q);
	  }
	}
      }
    }

    //---------------------------------
    mod->markerID++;
    cod_obj = mod->GetCodeValue(mod->markerID, o+1);
    APP->SetLabelColour(cod_obj, mod->obj_color[o]);

    mod->markerID++;
    cod_bkg = mod->GetCodeValue(mod->markerID, 0);
    APP->SetLabelColour(cod_bkg, NIL);

    for(a.y=0; a.y<tmp->nrows; a.y++){
      for(a.x=0; a.x<tmp->ncols; a.x++){
	p = a.x + a.y*tmp->ncols;
	if(tmp->val[p]==1 ||
	   tmp->val[p]==0){
	  u = GetCorrespondingVoxel(p);
	  q = VoxelAddress(APP->Data.orig,u.x,u.y,u.z);
	  if(tmp->val[p]==1){
	    APP->AddSeed(q, o+1, mod->markerID-1);
	    (APP->Data.label)->data[q] = cod_obj;
	  }
	  else if(tmp->val[p]==0){
	    APP->AddSeed(q, 0, mod->markerID);
	    (APP->Data.label)->data[q] = cod_bkg;
	  }
	}
      }
    }

    FIFOQDestroy(Q);
    DestroyAdjRel(&A8);
    DestroyAdjRel(&A4);
    DestroyAdjRel(&R);
    DestroyImage(&tmp);
  }
   */

  bool LiveHandler :: IsAnchorPoint(int px){
    int i;
    for(i=0; i<npoints; i++)
      if(anchorpoints[i]==px)
	return true;
    return false;
  }

  int LiveHandler :: GetLastAnchorPoint(){
    if(npoints>0)
      return anchorpoints[npoints-1];
    else
      return NIL;
  }

  int LiveHandler :: GetFirstAnchorPoint(){
    if(npoints>0)
      return anchorpoints[0];
    else
      return NIL;
  }

  void LiveHandler :: AppendAnchorPoint(int px){
    anchorpoints[npoints] = px;
    npoints++;
  }

  void LiveHandler :: DelLastAnchorPoint(){
    if(npoints>0)
      npoints--;
  }

  void LiveHandler :: ClearAnchorPoints(){
    npoints = 0;
  }

} //end Interactive namespace

