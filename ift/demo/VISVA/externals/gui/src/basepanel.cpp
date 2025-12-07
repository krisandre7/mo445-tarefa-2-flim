
#include "basepanel.h"


BasePanel :: BasePanel(wxWindow *parent) 
  : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxNO_BORDER, _T("basepanel")){ //wxSIMPLE_BORDER
  //SetBackgroundColour(*wxWHITE);

  this->Hide();
  sizer = new wxBoxSizer(wxVERTICAL);
  this->SetSizer(sizer, true);
  sizer->SetSizeHints(this);
  sizer->Layout();
}


BasePanel :: ~BasePanel(){
  sizer->Clear(true);
}

