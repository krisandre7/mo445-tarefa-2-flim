
#include "spinslider.h"


SpinSlider :: SpinSlider(wxWindow* parent, wxWindowID id,
			 int value , int minValue, int maxValue)
  : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize,
	    wxTAB_TRAVERSAL, _T("ssliderpanel")){
  int v,ndigits=0;
  slider = new wxSlider(this, id, value, minValue, maxValue, 
			wxDefaultPosition, wxDefaultSize, 
			wxSL_HORIZONTAL|wxSL_LABELS, 
			wxDefaultValidator, _T("slider"));
  v = maxValue;
  while(v>0){
    ndigits++;
    v /= 10;
  }
  wxSize spsize(16*ndigits, -1);
  spin = new wxSpinCtrl(this, id, wxEmptyString, wxDefaultPosition, 
			spsize, wxSP_ARROW_KEYS|wxSP_VERTICAL,
			minValue, maxValue, value, _T("spin"));

  wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
  hsizer->Add(slider, 1, wxEXPAND);
  hsizer->Add(spin,   0, wxALIGN_RIGHT|wxALIGN_CENTRE);

  SetSizer(hsizer, true);
  hsizer->SetSizeHints(this);

  Connect( id, wxEVT_COMMAND_SPINCTRL_UPDATED,
	   wxSpinEventHandler(SpinSlider::OnSpin),
	   NULL, NULL );
  Connect( id, wxEVT_SCROLL_THUMBRELEASE,
	   wxScrollEventHandler(SpinSlider::OnScroll),
	   NULL, NULL );
  Connect( id, wxEVT_SCROLL_THUMBTRACK,
	   wxScrollEventHandler(SpinSlider::OnScroll),
	   NULL, NULL );
}


SpinSlider :: ~SpinSlider(){
  int id = slider->GetId();
  Disconnect( id, wxEVT_COMMAND_SPINCTRL_UPDATED,
	      wxSpinEventHandler(SpinSlider::OnSpin), 
	      NULL, NULL );
  Disconnect( id, wxEVT_SCROLL_THUMBRELEASE,
	      wxScrollEventHandler(SpinSlider::OnScroll), 
	      NULL, NULL );
  Disconnect( id, wxEVT_SCROLL_THUMBTRACK,
	      wxScrollEventHandler(SpinSlider::OnScroll), 
	      NULL, NULL );
}


void SpinSlider :: OnScroll(wxScrollEvent& event){
  int value = slider->GetValue();
  SetValue(value);
  Refresh(false, NULL);
  Update();
  event.Skip();
}

void SpinSlider :: OnSpin(wxSpinEvent& event){
  int value = spin->GetValue();  
  int id = slider->GetId();
  SetValue(value);
  Refresh(false, NULL);
  Update();
  wxScrollEvent newevent( wxEVT_SCROLL_THUMBTRACK, id);
  newevent.SetEventObject( this );
  // Send it
  GetEventHandler()->ProcessEvent( newevent );
}

int  SpinSlider :: GetMax(){
  return slider->GetMax();
}

int  SpinSlider :: GetMin(){
  return slider->GetMin();
}

int  SpinSlider :: GetValue(){
  return slider->GetValue();
}

void SpinSlider :: SetValue(int value){
  slider->SetValue(value);
  spin->SetValue(value);
}

void SpinSlider :: SetRange(int minValue, int maxValue){
  slider->SetRange(minValue, maxValue);
  spin->SetRange(minValue, maxValue);
}


