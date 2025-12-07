
#include "colourbutton.h"


ColourButton :: ColourButton(wxWindow *parent,
			     wxWindowID id)
  : wxBitmapButton(parent, id, wxNullBitmap,
		   wxDefaultPosition, wxDefaultSize,
		   wxBU_AUTODRAW, wxDefaultValidator,
		   _T("btbut")){

  SetValue(0xFFFF00);

  Connect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	   wxCommandEventHandler(ColourButton::OnPress),
	   NULL, NULL );
}


ColourButton :: ~ColourButton(){
  int id = GetId();

  Disconnect( id, wxEVT_COMMAND_BUTTON_CLICKED,
	      wxCommandEventHandler(ColourButton::OnPress),
	      NULL, NULL );
}


void ColourButton :: OnPress(wxCommandEvent& event){
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
  event.Skip();
}


int ColourButton :: GetValue(){
  return color;
}


void ColourButton :: SetValue(int color){
  this->color = color;
  
  wxImage wximg(16, 16, true);
  wxRect  rect(0, 0, 16, 16);
  wximg.SetRGB(rect, t0(color), t1(color), t2(color));
  wxBitmap bmColor(wximg, -1);
  this->SetBitmapLabel(bmColor);  
}

