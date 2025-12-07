
#include "floatctrl.h"


FloatCtrl::FloatCtrl(wxWindow* parent, wxWindowID id, 
		     long style)
  : wxTextCtrl(parent, id, _T("0.0"), wxDefaultPosition, wxDefaultSize, style, wxDefaultValidator, _T("floatCtrl")){
}

FloatCtrl::~FloatCtrl(){}


void FloatCtrl::OnChar(wxKeyEvent& event){
  int code = event.GetKeyCode();

  if( isdigit(code)      || 
      code==WXK_BACK     ||
      code==WXK_RETURN   ||
      code==WXK_DELETE   ||
      code==WXK_END      ||
      code==WXK_HOME     ||
      code==WXK_LEFT     ||
      code==WXK_RIGHT    ||
      code==WXK_SUBTRACT ||
      code==WXK_DECIMAL  ||
      code=='.'          ||
      code=='-' ){
    // key code is within legal range. 
    // we call event.Skip() so the
    // event can be processed either in 
    // the base wxWidgets class or the 
    // native control.
    event.Skip();
  }
  else{
    // illegal key hit. we don't call 
    // event.Skip() so the event is not 
    // processed anywhere else.
    wxBell();
  }
}


