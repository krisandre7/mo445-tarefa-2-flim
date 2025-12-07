

class Report {
public:
  Report();
  void addMarkerEvent(int id, int nseeds, int label);
  void addRunEvent();
  void write(char *filename);
  void clear();

private:
  char *eventLog[4096];
  int   n;
  int   id,nseeds,label;
};


