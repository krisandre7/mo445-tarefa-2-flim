
#include "brushpicker.h"

#include "../xpm/arrowdown.xpm"
#include "../xpm/arrowup.xpm"


BrushPicker :: BrushPicker(wxWindow *parent,
			   wxWindowID id, bool initdefault)
  : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxBORDER_SUNKEN, _T("bpicker")){
  AdjRel *brush=NULL;
  int i;
  owner = parent;
  this->n = 0; 
  maxWidth = maxHeight = 0;
  this->brushes = NULL;
  
  scanvas = new SimpleCanvas(this);

  if(initdefault){
    for(i=0; i<16; i++){
      brush = FastCircular((float)i);
      AddBrush(brush);
      DestroyAdjRel(&brush);
    }
    selection = 3;
  }
  else
    selection = 1;

  Refresh();

  wxBitmap *bmUp = new wxBitmap(arrowup_xpm);
  wxBitmap *bmDown = new wxBitmap(arrowdown_xpm);

  up = new wxBitmapButton(this, id, *bmUp, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_up"));
  down = new wxBitmapButton(this, id, *bmDown, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_down"));

  wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
  hsizer->Add(scanvas, 0, wxEXPAND);

  wxBoxSizer *vsizer = new wxBoxSizer(wxVERTICAL);

  vsizer->Add(up, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
  vsizer->AddSpacer(2);
  vsizer->Add(down, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

  hsizer->Add(vsizer, 0, wxALIGN_LEFT);

  SetSizer(hsizer, true);
  hsizer->SetSizeHints(this);

  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(BrushPicker::OnPress),
	   NULL, NULL );
}


BrushPicker :: ~BrushPicker(){
  int i,id = up->GetId();

  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(BrushPicker::OnPress),
	      NULL, NULL );
  for(i=0; i<this->n; i++)
    DestroyAdjRel(&brushes[i]);

  free(brushes);
}


void BrushPicker :: AddBrush(AdjRel *brush){
  Image *img = AdjRel2Image(brush);
  int w,h;

  this->n++;
  brushes = (AdjRel **)realloc(brushes, 
			       this->n*sizeof(AdjRel *));
  brushes[this->n-1] = CloneAdjRel(brush);

  h = img->nrows;
  w = img->ncols;
  if(w > this->maxWidth)
    this->maxWidth = w;
  if(h > this->maxHeight)
    this->maxHeight = h;
  scanvas->SetSize(this->maxWidth+2,
		   this->maxHeight+2);
  DestroyImage(&img);
}


void BrushPicker :: Refresh(){
  CImage *cimg=NULL;
  Image *img=NULL,*sub=NULL;
  int p,size;

  if(this->n>0){
    img = CreateImage(this->maxWidth, 
		      this->maxHeight);
    sub = AdjRel2Image(brushes[selection]);
    PasteSubImage(img, sub, 
		  img->ncols/2-sub->ncols/2, 
		  img->nrows/2-sub->nrows/2);

    size = img->ncols*img->nrows;
    for(p=0; p<size; p++){
      if(img->val[p]==0)
	img->val[p]=255;
      else
	img->val[p]=0;
    }
    cimg = Convert2CImage(img);
    scanvas->DrawCImage(cimg);
    DestroyImage(&sub);
    DestroyImage(&img);
    DestroyCImage(&cimg);
  }
  scanvas->Refresh();
}


AdjRel *BrushPicker :: GetBrush(){
  if(n==0) return NULL;
  return (CloneAdjRel(brushes[selection]));
}


void BrushPicker :: OnPress(wxCommandEvent & event){
  if(event.GetEventObject()==up)
    this->NextBrush();
  else
    this->PrevBrush();
}


void BrushPicker :: NextBrush(){
  if(selection<n-1){
    selection++;
    Refresh();
  }
}

void BrushPicker :: PrevBrush(){
  if(selection>0){
    selection--;
    Refresh();
  }
}


wxCursor *BrushPicker :: GetBrushCursor(int zoom){
  Image *bin;
  Image *border;
  AdjRel *A;
  int ncols,nrows;
  wxCursor *cursor=NULL;

  A = this->GetBrush();
  bin = AdjRel2Image(A);
  border = ObjectBorder(bin);

  ncols = border->ncols;
  nrows = border->nrows;
  if(ncols*nrows==1) border->val[0] = 1;

  cursor = Image2Cursor(border, border,
			ncols/2, nrows/2, 
		        zoom, 0xFFFFFF, 0x000000);

  DestroyImage(&bin);
  DestroyImage(&border);
  DestroyAdjRel(&A);

  return cursor;
}




