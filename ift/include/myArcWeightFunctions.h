#ifndef ARC_WEIGHT_FUNCTIONS_H
#define ARC_WEIGHT_FUNCTIONS_H

typedef struct myDynTrees_ myDynTrees;
typedef float (*iftArcWeightFn)(int p, int q, int r, myDynTrees *dt);

typedef struct ArcWEntry {
    int id;
    iftArcWeightFn fn;
} ArcWEntry;

ArcWEntry* getArcWEntry(int id);
int getArcWTableSize(void);
ArcWEntry* getArcWTable(void);

float iftArcWeight1(int p, int q, int r, myDynTrees *dt);
float iftArcWeight2(int p, int q, int r, myDynTrees *dt);
float iftArcWeight3(int p, int q, int r, myDynTrees *dt);
float iftArcWeight4(int p, int q, int r, myDynTrees *dt);
float iftArcWeight5(int p, int q, int r, myDynTrees *dt);
float iftArcWeight6(int p, int q, int r, myDynTrees *dt);
#endif