



typedef struct {
  char name[12];
  float value;
} measure_node;



typedef struct {
  int n;
  measure_node *measure;  // array
} Measures;



Measures* CreateMeasures();
void DestroyMeasures(Measures **M);
void AddMeasure(Measures *M, char *name, float value);
int SearchMeasureByName(Measures *M, char *name, float *value);
void PrintMeasuresList(Measures *M);



void ComputeM1(Measures *M, Scene *scn);
void ComputeM2(Measures *M, Scene *scn);
