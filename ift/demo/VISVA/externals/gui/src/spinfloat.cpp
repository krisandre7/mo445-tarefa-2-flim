
#include "spinfloat.h"

#include "../xpm/arrowdown.xpm"
#include "../xpm/arrowup.xpm"


SpinFloat :: SpinFloat(wxWindow* parent, wxWindowID id,
		       float min, float max, float initial,
		       float increment, int digits)
  : wxPanel(parent, id, wxDefaultPosition, wxDefaultSize,
	    wxTAB_TRAVERSAL, _T("sfloatpanel")){
  this->value = initial;
  this->minValue = min;
  this->maxValue = max;
  this->increment = increment;
  this->digits = digits;

  wxBitmap *bmUp = new wxBitmap(arrowup_xpm);
  wxBitmap *bmDown = new wxBitmap(arrowdown_xpm);

  up = new wxBitmapButton(this, id, *bmUp, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_up"));
  down = new wxBitmapButton(this, id, *bmDown, wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW, wxDefaultValidator, _T("b_down"));

  ctrl = new FloatCtrl(this, id, wxTE_RIGHT);
  SetValue(initial);

  wxBoxSizer *hsizer = new wxBoxSizer(wxHORIZONTAL);
  hsizer->Add(ctrl, 0, wxEXPAND);

  wxBoxSizer *vsizer = new wxBoxSizer(wxVERTICAL);
  vsizer->Add(up, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
  vsizer->Add(down, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);

  hsizer->Add(vsizer, 0, wxALIGN_LEFT);

  SetSizer(hsizer, true);
  hsizer->SetSizeHints(this);

  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(SpinFloat::OnPress),
	   NULL, NULL );
  Connect( id, wxEVT_COMMAND_TEXT_UPDATED,
	   wxCommandEventHandler(SpinFloat::OnChange),
	   NULL, NULL );
  Connect( id, wxEVT_COMMAND_TEXT_ENTER,
	   wxCommandEventHandler(SpinFloat::OnEnter),
	   NULL, NULL );
}


SpinFloat :: ~SpinFloat(){
  int id = up->GetId();
  
  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(SpinFloat::OnPress),
	      NULL, NULL );
  Disconnect( id, wxEVT_COMMAND_TEXT_UPDATED,
	      wxCommandEventHandler(SpinFloat::OnChange),
	      NULL, NULL );
  Disconnect( id, wxEVT_COMMAND_TEXT_ENTER,
	      wxCommandEventHandler(SpinFloat::OnEnter),
	      NULL, NULL );
}


float SpinFloat::GetMax(){
  return this->maxValue;
}

float SpinFloat::GetMin(){
  return this->minValue;
}

float SpinFloat::GetValue(){
  ValidateContents();
  return this->value;
}

void  SpinFloat::SetValue(float value){
  char str[512];
  char format[100];
  int i;

  if(value>maxValue)
    value = maxValue;
  else if(value<minValue)
    value = minValue;

  this->value = value;
  sprintf(format,"%%.%df",this->digits);
  sprintf(str,format,value);
  for(i=0; i<(int)strlen(str); i++)
    if(str[i]==',') str[i]='.';
  wxString wxstr(str, wxConvUTF8);
#if wxMINOR_VERSION>=8
  ctrl->ChangeValue(wxstr);
#else
  ctrl->SetValue(wxstr);
#endif
}

void  SpinFloat::SetRange(float minValue, float maxValue){
  this->minValue = minValue;
  this->maxValue = maxValue;

  if(value>maxValue)
    SetValue(maxValue);
  else if(value<minValue)
    SetValue(minValue);
}

void  SpinFloat::OnPress(wxCommandEvent& event){
  ValidateContents();

  if(event.GetEventObject()==up){
    value += increment;
    if(value>maxValue) 
      value = maxValue;
  }
  else if(event.GetEventObject()==down){
    value -= increment;
    if(value<minValue) 
      value = minValue;
  }
  SetValue(value);

  //Send Event:
  wxSpinEvent evt( wxEVT_COMMAND_SPINCTRL_UPDATED, GetId() );
  evt.SetEventObject( this );
  GetEventHandler()->ProcessEvent( evt );
}

void  SpinFloat::OnChange(wxCommandEvent& event){
  //Send Event:
  wxSpinEvent evt( wxEVT_COMMAND_SPINCTRL_UPDATED, GetId() );
  evt.SetEventObject( this );
  GetEventHandler()->ProcessEvent( evt );
}

void  SpinFloat::OnEnter(wxCommandEvent& event){
  ValidateContents();
  //Send Event:
  wxSpinEvent evt( wxEVT_COMMAND_SPINCTRL_UPDATED, GetId() );
  evt.SetEventObject( this );
  GetEventHandler()->ProcessEvent( evt );
}


void SpinFloat::ValidateContents(){
  char str[512];
  char fstr[512];
  char *pstr;
  int i,j,flag=0;

  wxString wxstr = ctrl->GetValue();
  strcpy( str, wxstr.ToAscii() );
  for(i=0; i<(int)strlen(str); i++)
    if(str[i]==',') str[i]='.';
  j = 0;
  for(i=0; i<(int)strlen(str); i++){
    if(str[i]=='.'){ 
      if(!flag) flag=1;
      else continue;
    }
    if((str[i]>='0' && str[i]<='9')||
       (str[i]=='.')){
      fstr[j] = str[i];
      j++;
    }
  }
  fstr[j] = '\0';

  flag = 0;
  for(i=0; i<(int)strlen(fstr); i++){
    if(fstr[i]=='.'){ 
      j = i;
      flag = 1;
    }
  }

  if(!flag)
    value = (float)atoi(fstr);
  else{
    value = 0.0;
    pstr = &fstr[j+1];
    if(j<(int)strlen(fstr)-1) 
      value += (float)atoi(pstr)*pow(10.0,-1.0*strlen(pstr));

    fstr[j] = '\0';
    if(j!=0) value += (float)atoi(fstr);
  }
  SetValue(value);
}


