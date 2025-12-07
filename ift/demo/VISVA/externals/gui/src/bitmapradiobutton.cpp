
#include "bitmapradiobutton.h"

BitmapRadioButton :: BitmapRadioButton(wxWindow *parent,
				       wxWindowID id,
				       wxBitmap *bitmap[],
				       int n)
  : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxBORDER_NONE, _T("bradiobutton")){
  int i,w,h,hm=0;
  wxSize size;

  owner = parent;
  this->id = id;
  this->n = n;
  this->hbs = new wxBoxSizer(wxHORIZONTAL);
  v_but = new BitmapToggleButton*[n];
  for(i=0; i<n; i++){
    v_but[i] = new BitmapToggleButton(this, id, 
				      bitmap[i], bitmap[i]);
    v_but[i]->SetBkgColor(1, 0x929292);
    hbs->Add(v_but[i], 0, wxALIGN_CENTER|wxALIGN_CENTER_VERTICAL);

    v_but[i]->GetSize(&w, &h);
    if(h>hm) hm = h;
  }

  size.Set(hm, hm);
  for(i=0; i<n; i++){
    v_but[i]->SetSize(wxDefaultCoord, hm);
    v_but[i]->SetMinSize(size);
  }

  SetSizer(hbs, true);
  hbs->SetSizeHints(this);
  hbs->Layout();

  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(BitmapRadioButton::OnPress),
	   NULL, NULL );
}



BitmapRadioButton :: ~BitmapRadioButton(){
  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(BitmapRadioButton::OnPress),
	      NULL, NULL );
  delete v_but;
}


void BitmapRadioButton :: OnPress(wxCommandEvent& event){
  int i;

  for(i=0; i<n; i++)
    if(event.GetEventObject()!=v_but[i])
      v_but[i]->SetValue(false);

  event.Skip();
}


int BitmapRadioButton :: GetSelection(){
  int i;
  
  for(i=0; i<n; i++)
    if(v_but[i]->GetValue())
      break;

  if(i==n) return wxNOT_FOUND;
  else     return i;
}


void BitmapRadioButton :: SetSelection(int sel){
  int i;

  for(i=0; i<n; i++){
    if(i==sel)
      v_but[i]->SetValue(true);
    else
      v_but[i]->SetValue(false);
  }
}


void BitmapRadioButton :: DisableButton(int pos){
  if(pos>=n || pos<0) return;

  v_but[pos]->SetValue(false);
  v_but[pos]->Hide();
  hbs->Layout();
}


void BitmapRadioButton :: EnableButton(int pos){
  if(pos>=n || pos<0) return;

  v_but[pos]->Show(true);
  hbs->Layout();
}


