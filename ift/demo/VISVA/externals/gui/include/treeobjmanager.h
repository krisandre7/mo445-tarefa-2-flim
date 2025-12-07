


class TreeObjManager{
private:
  wxDialog     *dialog;
  wxStaticText *dlgText;
  wxBoxSizer   *vsizer;
  wxTreeItemId *Label2Id;
  void          ReadObjectTree(XMLDocument *doc, wxWindow *parent, GenSet **ObjID);
  void          ReadObjectNode(XMLNode *node, wxTreeItemId pred_id, GenSet **ObjID);

public:
  wxTreeCtrl   *tree;
  TreeObjManager(wxWindow *parent, char *filename);
  TreeObjData *GetObjDataByLabel(int lb);
  void ShowWindow();
  void ShowAcceptDialog();
};



