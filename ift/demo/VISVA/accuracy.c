#include "oldift.h"

int main(int argc, char **argv) {

  int TO=0,TB=0,FP=0,FN=0,N=0;
  Scene *A,*B;
  int i;
  float RFP,RFN,RE;

  if (argc!=3) {
    printf("usage: accuracy truth-scn test-scn\n\n");
    return 1;
  }

  A = ReadScene(argv[1]);
  B = ReadScene(argv[2]);

  if (A->xsize != B->xsize ||
      A->ysize != B->ysize ||
      A->zsize != B->zsize) {
    printf("scenes are uncomparable.\n");
    return 1;
  }

  N = A->xsize * A->ysize * A->zsize;

  for(i=0;i<N;i++) {
    if (A->data[i] == 0) {
      TB++;
      if (B->data[i]!=0) FP++;
    } else {
      TO++;
      if (B->data[i]==0) FN++;
    }
  }

  RFP = ((float)FP)/((float)TB);
  RFN = ((float)FN)/((float)TO);
  RE = ((float)(FN+FP)) / ((float)N);

  printf("accuracy: FP=%.4f FN=%.4f E=%.4f\n",RFP,RFN,RE);

  return 0;
}
