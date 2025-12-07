
#include "simplecanvas.h"


BEGIN_EVENT_TABLE(SimpleCanvas, wxPanel)
  EVT_PAINT(SimpleCanvas::OnPaint)
END_EVENT_TABLE()


SimpleCanvas :: SimpleCanvas(wxWindow *parent) 
    : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSIMPLE_BORDER, _T("scanvas"))
{
  owner = parent;
  ibuf = new wxMemoryDC;
  wximg = NULL;
  SetBackgroundColour(*wxWHITE);
  imageWidth = imageHeight = -1;
  loaded = false;
}


void SimpleCanvas :: DrawCImage(CImage *cimg){
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

  wxBitmap wxbit(*wximg, -1);
  ibuf->SelectObject(wxbit);
  loaded = true;
}


void SimpleCanvas :: DrawFigure(wxBitmap bitmap) {
  ibuf->SelectObject(bitmap);
  loaded = true;
}

void SimpleCanvas :: ClearFigure() {
  ibuf->SelectObject(wxNullBitmap);
  loaded = true;
}

void SimpleCanvas :: Refresh(){
  wxPanel::Refresh(true, NULL);
  Update();
}
  
void SimpleCanvas :: OnPaint(wxPaintEvent& event) {
  if(!loaded) return;

  wxSize size = ibuf->GetSize();
  imageWidth = size.GetWidth();
  imageHeight = size.GetHeight();

  wxPaintDC paintDC(this);
  //DoPrepareDC(paintDC);
  paintDC.Blit(0, 0, size.GetWidth(), size.GetHeight(), ibuf, 0, 0);
}


