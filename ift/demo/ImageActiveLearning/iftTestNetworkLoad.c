#include <ift.h>

int main(int argc, char* argv[])
{
  if (argc < 4) {
    printf("Usage %s <model.json> <weights> <test_img>\n", argv[0]);
    return -1;
  }

  timer *t1, *t2;
  
  t1 = iftTic();
  iftNeuralNetwork* net = iftCreateWeightedNeuralNetworkFromJson(argv[1], argv[2], NULL, 1, true); 
  t2 = iftToc();
  printf("Network load time: %fms\n", iftCompTime(t1, t2));

  int emptyBatchSlots = iftFeedEmptyImageLayerImgPath(net, argv[3]);
  (void) emptyBatchSlots; // Avoid warning

  t1 = iftTic();
  iftNetForward(net);
  t2 = iftToc();
  printf("Network forward time: %fms\n", iftCompTime(t1, t2));

  iftNeuralLayer *lastLayer = net->layers[net->nlayers - 1];
  //iftNeuralLayer *lastLayer = net->layers[19];
  //iftTensor *t = lastLayer->out;
  //iftTensor *www = net->layers[1]->weight;
  /*for (int dim2 = 0; dim2 < t->dimension[3]; ++dim2) {
    for (int dim1 = 0; dim1 < t->dimension[2]; ++dim1) {
      for (int dim0 = 0; dim0 < t->dimension[1]; ++dim0) {
        int index = dim0 * t->accDimension[1] + dim1 * t->accDimension[2] + dim2 * t->accDimension[3];
        //int index = unit * www->accDimension[0] + dim0 * t->accDimension[1] + dim1 * t->accDimension[2] + dim2 * t->accDimension[3];
        printf("kernel %d x %d y %d = val[%d] = %f\n", dim0, dim1, dim2, index, t->val[index]);
        //printf("unit %d band %d x %d y %d = val[%d] = %f\n", unit, dim0, dim1, dim2, index, www->val[index]);
      }
    }
  }*/
  for (int i = 0; i < lastLayer->out->n; ++i)
    printf("Output %d = %f\n", i, lastLayer->out->val[i]);
  
  iftDestroyNeuralNetwork(&net);

  return 0;
}
