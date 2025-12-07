
#include "canvas.h"


BEGIN_EVENT_TABLE(Canvas, wxScrolledWindow)
  EVT_PAINT(Canvas::OnPaint)
  EVT_MOUSE_EVENTS(Canvas::OnMouseEvent)
END_EVENT_TABLE()


Canvas :: Canvas(wxWindow *parent) 
  : wxScrolledWindow(parent, wxID_ANY, 
		     wxDefaultPosition, wxDefaultSize,
		     wxSIMPLE_BORDER|wxVSCROLL|wxHSCROLL){
  owner = parent;
  ibuf = new wxMemoryDC;  
  wximg = NULL;

  imageWidth = imageHeight = -1;
  displayWidth = displayHeight = -1;
  zoom = 1.0;
  handler = NULL;

  SetBackgroundColour(*wxBLACK);
}


Canvas :: ~Canvas(){
  if(wximg!=NULL)
    delete wximg;
}


void Canvas :: DrawCImage(CImage *cimg){
  int n,p,q,ncols,nrows;

  ncols = cimg->C[0]->ncols;
  nrows = cimg->C[0]->nrows;

  if(imageWidth!=ncols || imageHeight!=nrows){
    imageWidth  = ncols;
    imageHeight = nrows;
    if(wximg!=NULL)
      delete wximg;

    wximg = new wxImage(ncols, nrows, false);
  }

  unsigned char*data = wximg->GetData();
  n = ncols*nrows;
  for(p=0,q=0;p<n;p++,q+=3){
    data[q]   = cimg->C[0]->val[p];
    data[q+1] = cimg->C[1]->val[p];
    data[q+2] = cimg->C[2]->val[p];
  }

  wxImage tmp(*wximg);
  if(zoom!=1.0){
#if wxMINOR_VERSION>=8
    tmp.Rescale(ROUND(ncols*zoom), ROUND(nrows*zoom),
		wxIMAGE_QUALITY_NORMAL);
#else
    tmp.Rescale(ROUND(ncols*zoom), ROUND(nrows*zoom));
#endif
  }

  wxBitmap wxbit(tmp, -1);
  ibuf->SelectObject(wxbit);
}

void Canvas :: ClearFigure() {
  if(wximg != NULL){
    delete wximg;
    wximg = NULL;
  }
  ibuf->SelectObject(wxNullBitmap);
}


void Canvas :: Draw(){

}

//Returns the pixel address.
int  Canvas :: Canvas2Address(int x, int y){
  int scx,scy,xUnit,yUnit;
  int w,h;
  float fx,fy;

  GetViewStart(&scx, &scy);
  GetScrollPixelsPerUnit(&xUnit, &yUnit);
  scx *= xUnit;
  scy *= yUnit;
  x += scx;
  y += scy;

  fx = (float)x / this->zoom;
  fy = (float)y / this->zoom;
  x = ROUND(fx);
  y = ROUND(fy);

  w = this->imageWidth;
  h = this->imageHeight;
  // Só processa o evento caso o mouse esteja 
  //dentro da imagem.
  if(x<0 || y<0 || x>=w || y>=h) return -1;

  return (x+y*w);
}


void Canvas :: Refresh(){
  wxScrolledWindow::Refresh(true, NULL);
  Update();
}
  
void Canvas :: OnPaint(wxPaintEvent& event) {
  //if(!AppData.loaded) return;

  wxSize size = ibuf->GetSize();

  if(displayWidth!=size.GetWidth() || displayHeight!=size.GetHeight()){
    displayWidth = size.GetWidth();
    displayHeight = size.GetHeight();
    SetScrollRate(5, 5);
    Scroll(0, 0);
    SetVirtualSize(displayWidth,displayHeight);
  }

  wxPaintDC paintDC(this);
  DoPrepareDC(paintDC);
  paintDC.Blit(0, 0, size.GetWidth(), size.GetHeight(), ibuf, 0, 0);
}


void Canvas :: OnMouseEvent(wxMouseEvent& event){
  int p,x,y;
  
  event.GetPosition(&x, &y);
  p = Canvas2Address(x, y);
  if(handler!=NULL && p>=0)
    handler->OnMouseEvent(event, p);
}


void Canvas :: Zoomin(){
  if(zoom>=15.9999 || wximg==NULL) return;
  zoom *= 2.0;

  wxImage tmp(*wximg);
  if(zoom!=1.0){
#if wxMINOR_VERSION>=8
    tmp.Rescale(ROUND(imageWidth*zoom), 
		ROUND(imageHeight*zoom),
		wxIMAGE_QUALITY_NORMAL);
#else
    tmp.Rescale(ROUND(imageWidth*zoom), 
		ROUND(imageHeight*zoom));
#endif
  }

  wxBitmap wxbit(tmp, -1);
  ibuf->SelectObject(wxbit);

  this->Refresh();
}

void Canvas :: Zoomout(){
  if(zoom<=0.1250001 || wximg==NULL) return;
  zoom *= 0.5;

  wxImage tmp(*wximg);
  if(zoom!=1.0){
#if wxMINOR_VERSION>=8
    tmp.Rescale(ROUND(imageWidth*zoom), 
		ROUND(imageHeight*zoom),
		wxIMAGE_QUALITY_NORMAL);
#else
    tmp.Rescale(ROUND(imageWidth*zoom), 
		ROUND(imageHeight*zoom));
#endif
  }

  wxBitmap wxbit(tmp, -1);
  ibuf->SelectObject(wxbit);

  this->Refresh();
}


void Canvas :: SetZoomLevel(float zoom){
  if(zoom<=0.1250001 || zoom>=15.9999) return;

  this->zoom = zoom;

  if(wximg == NULL) return;
  wxImage tmp(*wximg);
  if(zoom!=1.0){
#if wxMINOR_VERSION>=8
    tmp.Rescale(ROUND(imageWidth*zoom), 
		ROUND(imageHeight*zoom),
		wxIMAGE_QUALITY_NORMAL);
#else
    tmp.Rescale(ROUND(imageWidth*zoom), 
		ROUND(imageHeight*zoom));
#endif
  }

  wxBitmap wxbit(tmp, -1);
  ibuf->SelectObject(wxbit);

  this->Refresh();
}


float Canvas :: GetZoomLevel(){
  return this->zoom;
}


void Canvas :: SetInteractionHandler(InteractionHandler *handler){
  this->handler = handler;
}

void Canvas :: GetImageSize(int* width, int* height) {
  *width = imageWidth;
  *height = imageHeight;
}


void Canvas :: GetDisplaySize(int* width, int* height) {
  *width = displayWidth;
  *height = displayHeight;
}


CImage * Canvas :: CopyAsCImage(){
  unsigned char *data;
  int n,p,q;
  CImage *cimg;

  if(imageWidth == -1 || imageHeight == -1 || wximg == NULL)
    return NULL;

  data = wximg->GetData();
  n = imageWidth*imageHeight;
  cimg = CreateCImage(imageWidth, imageHeight);
  for(p=0,q=0;p<n;p++,q+=3){
    cimg->C[0]->val[p] = data[q];
    cimg->C[1]->val[p] = data[q+1];
    cimg->C[2]->val[p] = data[q+2];
  }
  return cimg;
}


wxImage *Canvas :: GetImage() {
  return wximg;
}

