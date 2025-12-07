


typedef struct _appObjId{
  wxTreeItemId id;
} AppObjId;

TreeObjManager::TreeObjManager(wxWindow *parent, char *filename){
  wxTreeItemId id;
  AppObjId *appid;
  TreeObjData *obj;
  GenSet *ObjID=NULL,*S;
  int n, lb;

  dialog = new wxDialog(parent, -1, _T(""), wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER, _T("dialogBox"));

  vsizer = new wxBoxSizer(wxVERTICAL);

  dlgText = new wxStaticText(dialog, -1, _T(""), wxDefaultPosition, wxDefaultSize, wxALIGN_LEFT, _T("dlgtext"));

  wxStaticLine *sline1 = new wxStaticLine(dialog, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine1"));

  XMLDocument *doc = ReadXMLDocument(filename);
  ReadObjectTree(doc, dialog, &ObjID);

  n = tree->GetCount();
  Label2Id = (wxTreeItemId *)calloc(n, sizeof(wxTreeItemId));
  S = ObjID;
  while(S!=NULL){
    appid = (AppObjId *)S->elem;
    id = appid->id;
    if( !id.IsOk() )
      Error("Invalid id","ReadObjectTree");
    obj = (TreeObjData *)tree->GetItemData(id);
    lb  = obj->GetLabel();
    Label2Id[lb] = id;
    S = S->next;
  }
  DestroyXMLDocument(&doc);
  DestroyGenSet(&ObjID);

  wxStaticLine *sline2 = new wxStaticLine(dialog, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL, _T("staticLine2"));
  wxSizer *sbutton = dialog->CreateButtonSizer(wxOK|wxCANCEL);

  vsizer->AddSpacer(10);
  vsizer->Add(dlgText, 0, wxALIGN_LEFT|wxALIGN_CENTER_VERTICAL);
  vsizer->AddSpacer(10);
  vsizer->Add(sline1,  0, wxEXPAND);
  vsizer->Add(tree,    1, wxEXPAND);
  vsizer->Add(sline2,  0, wxEXPAND);
  vsizer->Add(sbutton, 0, wxEXPAND); //wxALIGN_LEFT|wxEXPAND
  dialog->SetSizer(vsizer, true);
  vsizer->SetSizeHints(dialog);
  vsizer->Layout();
}


void TreeObjManager::ReadObjectTree(XMLDocument *doc, wxWindow *parent, GenSet **ObjID){
  wxTreeItemId id,rid;
  AppObjId *appid;
  TreeObjData *obj;
  XMLNode *node,*root,*first;
  wxBitmap accept(smallaccept_xpm);
  wxBitmap notaccept(smallnotaccept_xpm);
  GenSet *S;
  
  tree = new wxTreeCtrl(parent, -1, wxDefaultPosition, wxDefaultSize, wxTR_HAS_BUTTONS, wxDefaultValidator, _T("listCtrl"));
  wxImageList *imgList = new wxImageList();
  imgList->Add(notaccept, wxNullBitmap);
  imgList->Add(accept,    wxNullBitmap);
  tree->AssignImageList(imgList);
  rid = tree->AddRoot(_T("Volume"), -1, -1, new TreeObjData("background"));

  root = doc->root;
  if(root->type!=TAG || strcasecmp((root->contents).tag.name,"objtree")!=0)
    Error("Invalid action!","ReadObjectTree");

  node = (root->contents).tag.childList;
  if(node!=NULL){
    first = node = node->next;
    do{
      ReadObjectNode(node, rid, ObjID);
      node = node->next;
    }while(node!=first);
  }

  S = *ObjID;
  while(S!=NULL){
    appid = (AppObjId *)S->elem;
    id = appid->id;
    if( !id.IsOk() )
      Error("Invalid id","ReadObjectTree");
    obj = (TreeObjData *)tree->GetItemData(id);
    obj->Unaccept(tree);
    tree->Expand(id);
    S = S->next;
  }
  obj = (TreeObjData *)tree->GetItemData(rid);
  obj->Accept(tree);
  tree->Expand(rid);
  appid = (AppObjId *)calloc(1, sizeof(AppObjId));
  appid->id = rid;
  InsertGenSet(ObjID, appid);
}


void TreeObjManager::ReadObjectNode(XMLNode *node, wxTreeItemId pred_id, GenSet **ObjID){
  XMLTagAttrib *attrList;
  wxTreeItemId id;
  AppObjId *appid;
  XMLNode *first;
  char *name=NULL;

  if(node==NULL)
    return; 
  
  switch(node->type){
  case TAG:
    if(strcasecmp((node->contents).tag.name,"obj")==0){
      attrList = node->contents.tag.attrList;
      while(attrList!=NULL){
	if(strcasecmp(attrList->name,"name")==0)
	  name = attrList->value;
	attrList = attrList->next;
      }
      if(name==NULL)
	Error("Invalid obj: missing 'name'!","ReadObjectNode");

      wxString wxname(name, wxConvUTF8);
      id = tree->AppendItem(pred_id, wxname, -1, -1, new TreeObjData(name));
      appid = (AppObjId *)calloc(1, sizeof(AppObjId));
      appid->id = id;
      InsertGenSet(ObjID, appid);
      node = (node->contents).tag.childList;
      if(node!=NULL){
	first = node = node->next;
	do{
	  ReadObjectNode(node, id, ObjID);
	  node = node->next;
	}while(node!=first);
      }
    }
    else
      Error("Invalid TAG!","ReadObjectNode");
    break;
  default:
    break;
  }
}

TreeObjData *TreeObjManager::GetObjDataByLabel(int lb){
  wxTreeItemId id;
  int n = tree->GetCount();

  if(lb<n){
    id = Label2Id[lb];
    if( id.IsOk() )
      return (TreeObjData *)tree->GetItemData(id);
  }
  return NULL;
}


void TreeObjManager::ShowWindow(){
  dialog->SetTitle(_T("Structure Manager"));
  dlgText->SetLabel(_T("   Select a structure for advanced options:   "));
  vsizer->SetSizeHints(dialog);
  vsizer->Layout();
  dialog->Show(true);
}


void TreeObjManager::ShowAcceptDialog(){
  int lb,p,n;
  wxWindowDisabler disableAll(dialog);
  wxTheApp->Yield();
  
  if(dialog->IsModal())
    dialog->EndModal(wxID_CANCEL);
  else
    dialog->Show(false);

  dialog->SetTitle(_T("Accept Segmentation"));
  dlgText->SetLabel(_T("   Choose the structure to which you wish to assign   \n   the current segmentation.   "));
  vsizer->SetSizeHints(dialog);
  vsizer->Layout();
  if(dialog->ShowModal()==wxID_OK){
    wxTreeItemId id  = tree->GetSelection();    
    if( id.IsOk() ){
      TreeObjData *obj = (TreeObjData *)tree->GetItemData(id);
      obj->Accept(tree);
      lb = obj->GetLabel();
      n = AppData.w*AppData.h*AppData.nframes;
      for(p=0; p<n; p++){
	if( GetMarkerLabel(AppData.mark->data[p]) > 0 )
	  AppData.flabel->data[p] = lb;
      }
      wxMessageBox(_T("Operation completed successfully!"), 
		   _T("Success"), wxOK | wxICON_EXCLAMATION, Window);
    }
    else
      wxMessageBox(_T("Wrong selection!"), 
		   _T("Error"), wxOK | wxICON_EXCLAMATION, Window);
  }
}


