


int           TreeObjData::n = 0;


TreeObjData::TreeObjData(char *name) : wxTreeItemData(){
  strcpy(this->name, name);
  accepted = false;
  label = n;
  n++;
}

TreeObjData::~TreeObjData(){ 
  n--; 
}


int TreeObjData::GetLabel(){
  return label;
}

char *TreeObjData::GetName(){
  return name;
}

bool TreeObjData::IsAccepted(){
  return accepted;
}

void TreeObjData::Accept(wxTreeCtrl *tree){
  accepted = true;
  wxTreeItemId id = GetId();
  if( !id.IsOk() )
    Error("wxTreeItemId not valid","Accept");
  tree->SetItemImage(id, 1, wxTreeItemIcon_Normal);
  tree->SetItemTextColour(id, *wxBLACK);
  tree->SetItemBold(id, true);
}

void TreeObjData::Unaccept(wxTreeCtrl *tree){
  accepted = false;
  wxTreeItemId id = GetId();
  if( !id.IsOk() )
    Error("wxTreeItemId not valid","Unaccept");
  tree->SetItemImage(id, 0, wxTreeItemIcon_Normal);
  tree->SetItemTextColour(id, *wxRED);
  tree->SetItemBold(id, false);
}


