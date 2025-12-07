
#include "alphacolourbutton.h"

#include "../xpm/arrowdown.xpm"
#include "../xpm/arrowup.xpm"


AlphaColourButton :: AlphaColourButton(wxWindow *parent,
				       wxWindowID id)
  : wxPanel(parent, id, wxDefaultPosition, wxDefaultSize, wxBORDER_SUNKEN, _T("acolourbutton")){
  owner = parent;
  if(id==wxID_ANY) id = this->GetId();
  
  wxBitmap *bmUp = new wxBitmap(arrowup_xpm);
  wxBitmap *bmDown = new wxBitmap(arrowdown_xpm);

  up = new wxBitmapButton(this, id, *bmUp, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_up"));
  down = new wxBitmapButton(this, id, *bmDown, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_down"));
  but = new wxBitmapButton(this, id, wxNullBitmap, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_color"));

  this->samples = AllocIntArray(16*16);
  this->alpha = 255;
  this->color = 0xFFFF00;
  this->SetAlpha(this->alpha);

  wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
  hsizer->Add(but, 0, wxEXPAND);

  wxBoxSizer *vsizer = new wxBoxSizer(wxVERTICAL);

  vsizer->Add(up, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
  //vsizer->AddSpacer(2);
  vsizer->Add(down, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

  hsizer->Add(vsizer, 0, wxALIGN_LEFT);

  SetSizer(hsizer, true);
  hsizer->SetSizeHints(this);

  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(AlphaColourButton::OnPress),
	   NULL, NULL );
}


AlphaColourButton :: ~AlphaColourButton(){
  int id = up->GetId();

  free(this->samples);
  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(AlphaColourButton::OnPress),
	      NULL, NULL );
}

void AlphaColourButton :: OnPress(wxCommandEvent& event){

  if(event.GetEventObject()==up){
    alpha += 10;
    if(alpha>255)
      alpha = 255;
    ChangeAlpha();
  }
  else if(event.GetEventObject()==down){
    alpha -= 10;
    if(alpha<0)
      alpha = 0;
    ChangeAlpha();
  }
  else{
    wxColourDialog *colourDialog;
    wxColourData cdata;
    wxColour c;
    int r,g,b;

    r = t0(this->color);
    g = t1(this->color);
    b = t2(this->color);
    c.Set(r,g,b);
    cdata.SetColour(c);
    colourDialog = new wxColourDialog(this, &cdata);
    if(colourDialog->ShowModal() == wxID_OK){
      cdata = colourDialog->GetColourData();
      c = cdata.GetColour();
      r = c.Red();
      g = c.Green();
      b = c.Blue();
      SetValue(triplet(r,g,b));
    }
  }
  event.Skip();
}

void AlphaColourButton :: Refresh(){
  int  maskcolor;
  int n,p;
  unsigned char *data;

  n = 16*16;
  if(color!=0x000000)
    maskcolor = 0x000000;
  else
    maskcolor = 0xffffff;

  wxImage wximg(16, 16, true);

  wximg.SetMaskColour(t0(maskcolor),
		      t1(maskcolor),
		      t2(maskcolor));

  wxRect  rect(0, 0, 16, 16);
  wximg.SetRGB(rect,
	       t0(maskcolor),
	       t1(maskcolor),
	       t2(maskcolor));

  data = wximg.GetData();
  for(p=0; p<n; p++){
    if(samples[p]>0){
      data[p*3]   = t0(color);
      data[p*3+1] = t1(color);
      data[p*3+2] = t2(color);
    }
  }

  wxBitmap bmColor(wximg, -1);
  this->but->SetBitmapLabel(bmColor);
}


int AlphaColourButton :: GetValue(){
  return color;
}

void AlphaColourButton :: SetValue(int color){
  this->color = color;
  this->Refresh();
}

int AlphaColourButton :: GetAlpha(){
  return alpha;
}

void AlphaColourButton :: SetAlpha(int alpha){
  int nsamples,p,q,n;

  this->alpha = alpha;
  n = 16*16;
  nsamples = ROUND((alpha/255.0)*n);
  for(p=0; p<nsamples; p++)
    samples[p] = 1;
  for(p=nsamples; p<n; p++)
    samples[p] = 0;

  srand((int)time(NULL));
  for(p=0; p<n; p++){
    q = RandomInteger(0, n-1);
    Change(&samples[p], &samples[q]);
  }

  this->Refresh();
}


void AlphaColourButton :: ChangeAlpha(){
  int n,p,q,s,nsamples=0,dsamples;

  n = 16*16;
  for(p=0; p<n; p++)
    if(samples[p]>0)
      nsamples++;

  dsamples = ROUND((alpha/255.0)*n) - nsamples;

  if(dsamples==0) return;
  else if(dsamples>0){
    while(dsamples>0){
      s = RandomInteger(0, n-1);
      p = s;
      do{
	if(samples[p]==0){
	  samples[p] = 1;
	  q = RandomInteger(0, n-1);
	  Change(&samples[p], &samples[q]);
	  break;
	}
	p = (p+1)%n;
      }while(p!=s);
      dsamples--;
    }
  }
  else{
    while(dsamples<0){
      s = RandomInteger(0, n-1);
      p = s;
      do{
	if(samples[p]==1){
	  samples[p] = 0;
	  q = RandomInteger(0, n-1);
	  Change(&samples[p], &samples[q]);
	  break;
	}
	p = (p+1)%n;
      }while(p!=s);
      dsamples++;
    }
  }

  this->Refresh();
}



