#include "ift.h"

int main(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: %s <InputZtrain.zip> <OutputCplGraph.zip>\n", argv[0]);
    exit(-1);
  }

  iftDataSet *Ztrain = iftReadDataSet(argv[1]);
  iftSetStatus(Ztrain, IFT_TRAIN);
  iftCplGraph *graph = iftCreateCplGraph(Ztrain);
  printf("Comp dist table...\n");
  iftDist = iftCompDistanceTable(Ztrain, Ztrain);
  printf("Sup train...\n");
  iftSupTrain(graph);
  iftDestroyDistanceTable(&iftDist);
  iftWriteCplGraph(graph, argv[2]);

  iftDestroyDataSet(&Ztrain);
  iftDestroyCplGraph(&graph);

  return 0;
}
