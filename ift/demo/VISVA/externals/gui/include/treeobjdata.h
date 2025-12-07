


class TreeObjData: public wxTreeItemData{
private:
  int label;
  bool accepted;
  char name[100];
  static int n;

public:
  int  size;
  TreeObjData(char *name);
  ~TreeObjData();
  int    GetLabel();
  char  *GetName();
  bool   IsAccepted();
  void   Accept(wxTreeCtrl *tree);
  void   Unaccept(wxTreeCtrl *tree);
};


