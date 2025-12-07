
#include "bitmaptogglebutton.h"


BitmapToggleButton :: BitmapToggleButton(wxWindow *parent,
					 wxWindowID id,
					 wxBitmap *bitmap0,
					 wxBitmap *bitmap1)
  : wxBitmapButton(parent, id, *bitmap0,
		   wxDefaultPosition, wxDefaultSize,
		   wxBU_AUTODRAW,wxDefaultValidator,_T("btbut")){
  state = false;
  //this->bitmap0 = bitmap0;
  //this->bitmap1 = bitmap1;
  this->bitmap0 = new wxBitmap(*bitmap0);
  this->bitmap1 = new wxBitmap(*bitmap1);

  this->color0  = 0xefebe7;
  this->color1  = 0xefebe7;

  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(BitmapToggleButton::OnPress),
	   NULL, NULL );
}

BitmapToggleButton :: ~BitmapToggleButton(){
  int id = GetId();

  //if(state) delete bitmap0;
  //else      delete bitmap1;
  delete bitmap0;
  delete bitmap1;
  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(BitmapToggleButton::OnPress),
	      NULL, NULL );
}

void BitmapToggleButton :: OnPress(wxCommandEvent& event){
  ToggleValue();
  event.Skip();
}

bool BitmapToggleButton :: GetValue(){
  return state;
}

void BitmapToggleButton :: SetValue(bool state){
  wxColour colour;
  SetColor(&colour, this->color1);
  this->state = state;
  if(state){
    SetBitmapLabel(*bitmap1);
    SetColor(&colour, this->color1);
    this->SetBackgroundColour(colour);
  }
  else{
    SetBitmapLabel(*bitmap0);
    SetColor(&colour, this->color0);
    this->SetBackgroundColour(colour);
  }
}


void BitmapToggleButton :: ToggleValue(){
  SetValue(!state);
}


void BitmapToggleButton :: SetBkgColor(bool state, int color){
  if(state)
    this->color1 = color;
  else
    this->color0 = color;
}

