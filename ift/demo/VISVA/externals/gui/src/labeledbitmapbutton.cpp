
#include "labeledbitmapbutton.h"


LabeledBitmapButton :: LabeledBitmapButton(wxWindow *parent,
					   wxWindowID id,
					     wxBitmap& bitmap,
					     wxString& label)
  : wxBitmapButton(parent, id, wxNullBitmap,
		   wxDefaultPosition, wxDefaultSize,
		   wxBU_AUTODRAW, 
		   wxDefaultValidator,
		   _T("btbut")){

  wxstr = new wxString(label);
  this->SetBitmapLabel(bitmap);
  /*
  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(LabeledBitmapButton::OnPress),
	   NULL, NULL );
  */
  this->SetSize(GetBestSize());
}


LabeledBitmapButton :: ~LabeledBitmapButton(){
  /*
  int id = GetId();

  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(LabeledBitmapButton::OnPress),
	      NULL, NULL );
  */
}


void LabeledBitmapButton::SetBitmapDisabled(  wxBitmap& bitmap){
  wxBitmap *bm;
  bm = this->LabelBitmap(bitmap, *wxstr);
  wxBitmapButton::SetBitmapDisabled(*bm);
}

void LabeledBitmapButton::SetBitmapFocus(  wxBitmap& bitmap){
  wxBitmap *bm;
  bm = this->LabelBitmap(bitmap, *wxstr);
  wxBitmapButton::SetBitmapFocus(*bm);
}

void LabeledBitmapButton::SetBitmapHover(  wxBitmap& bitmap){
#if wxMINOR_VERSION>=8
  wxBitmap *bm;
  bm = this->LabelBitmap(bitmap, *wxstr);
  wxBitmapButton::SetBitmapHover(*bm);
#endif
}

void LabeledBitmapButton::SetBitmapLabel(  wxBitmap& bitmap){
  wxBitmap *bm;

  //bitmap.SaveFile(_T("old.bmp"), wxBITMAP_TYPE_BMP, NULL);
  bm = this->LabelBitmap(bitmap, *wxstr);
  //bm->SaveFile(_T("new.bmp"), wxBITMAP_TYPE_BMP, NULL);
  wxBitmapButton::SetBitmapLabel(*bm);
}

void LabeledBitmapButton::SetBitmapSelected(  wxBitmap& bitmap){
  wxBitmap *bm;
  bm = this->LabelBitmap(bitmap, *wxstr);
  wxBitmapButton::SetBitmapSelected(*bm);
}

void LabeledBitmapButton::SetLabel(  wxString& label){
  delete wxstr;
  wxstr = new wxString(label);
  wxButton::SetLabel(label);
}


wxBitmap *LabeledBitmapButton::LabelBitmap(  wxBitmap& bitmap,
					   wxString& label){
  wxImage wximg;
  wxBitmap bitmap2(bitmap);
  wxMemoryDC ibuf,nibuf;
  wxSize  size,tsize;
  wxPoint point;
  int w,h,nw,nh,tw,th;
  unsigned char *data,*datastr;
  int n,p,q;

  ibuf.SelectObject(bitmap2);
  wxFont font(10,
	      wxFONTFAMILY_ROMAN, 
	      wxFONTSTYLE_NORMAL, 
	      wxFONTWEIGHT_NORMAL, 
	      false, _T("TimesRoman"),
	      wxFONTENCODING_DEFAULT);
  ibuf.SetFont(font);

#if wxMINOR_VERSION>=8
  tsize = ibuf.GetTextExtent(label);
#else
  ibuf.GetTextExtent(label, &w, &h,
		     NULL, NULL, NULL);
  tsize.Set(w, h);
#endif

  wximg = bitmap.ConvertToImage();

  w = wximg.GetWidth();
  h = wximg.GetHeight();

  tw = tsize.GetWidth();
  th = tsize.GetHeight();

  nw = MAX(tw+2,w);
  nh = th+h+2;
  size.Set(nw, nh);

  point.x = nw/2 - w/2;
  point.y = 0;

  wximg.Resize(size, point, -1,-1,-1);

  //printf("HasAlpha: %d, HasMask: %d\n",wximg.HasAlpha(),wximg.HasMask());

  wxImage wximgstr(nw, th+2, false);
  wxRect  rect(0, 0, nw, th+2);
  wximgstr.SetRGB(rect, 255, 255, 255);

  wxBitmap wxbit(wximgstr, -1);

  nibuf.SelectObject(wxbit);
  nibuf.SetFont(font);

  DrawWxString(&nibuf, &label,
	       nw/2, th/2+1,
	       0x000000, NIL, 0.5, 0.5, 0.0);

  wximgstr = wxbit.ConvertToImage();

  data = wximg.GetData();
  datastr = wximgstr.GetData();

  n = nw*(th+2);
  for(p=0; p<n; p++){
    q = p + nw*h;
    if(datastr[p*3]<200){
      data[q*3]   = datastr[p*3];
      data[q*3+1] = datastr[p*3+1];
      data[q*3+2] = datastr[p*3+2];
    }
  }

  wxBitmap *bm;
  bm = new wxBitmap(wximg, -1);

  return bm;
}


