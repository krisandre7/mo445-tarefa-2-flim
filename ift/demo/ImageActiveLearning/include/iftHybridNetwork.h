#ifndef _IFT_HYBRID_NETWORK_H_
#define _IFT_HYBRID_NETWORK_H_

#include <ift.h>

/**
 * @brief Load Vgg for the hybrid pipeline.
 * @date Oct 2019
 * @author Felipe Galvao
 *
 * @details Both json and weights should be obtained running the
 *   convertToIFTModel python script over the output .h5 file
 *   storing the keras trained network.
 *          Optimal choice of minibatch size depends on hardware, default = 1.
 *
 * @param json_path Json file containing the Vgg Neural Network architecture.
 * @param weights_path Binary file containing the Neural Network weights.
 * @param minibatch_size Number of samples processed simultaneously by the network.
 * @return Neural network
 */
iftNeuralNetwork* iftLoadVggFromJson(const char *json_path, const char *weights_path, int minibatch_size);

/**
 *  @brief Load image from path as input for the Vgg network.
 *  @author Felipe Galvao
 *  @date Nov, 2019
 *
 *  @details Similar to iftFeedEmptyLayerImgPath but adds Vgg specific initialization. 
 *    If mini_match_size > 1, each call loads image in the next batch
 *    free position. If batch is full (i.e., after this function returns 0),
 *    iftClearEmptyImageLayerBatch(net) must be called before calling this
 *    function again.
 *
 *  @param net Neural Network 
 *  @param img_path Image to be fed into the network
 *  @return Remaining slots in current batch
 */
int iftVggLoadImgFromPath(iftNeuralNetwork* net, const char *img_path);

/**
 *  @brief Load image from buffers as input for the Vgg network.
 *  @author Felipe Galvao
 *  @date Nov, 2019
 *
 *  @details Similar to iftFeedEmptyLayerColorImg but adds Vgg specific initialization. 
 *    If mini_match_size > 1, each call loads image in the next batch
 *    free position. If batch is full (i.e., after this function returns 0),
 *    iftClearEmptyImageLayerBatch(net) must be called before calling this
 *    function again.
 *
 *  @param net Neural Network 
 *  @param R Red channel
 *  @param G Green channel
 *  @param B Blue channel
 *  @return Remaining slots in current batch
 */
int iftVggLoadImgFromBuffer(iftNeuralNetwork *net, int *R, int *G, int *B);

/**
 *  @brief Classify a single sample with p-SVM. 
 *  @author Felipe Galvao
 *  @date Oct, 2019
 *
 *  @param svm Pre-trained SVM network
 *  @param feats Sample feature array
 *  @param nFeats Size of sample feature array
 *  @param outLabel Address to save SVM label prediction
 *  @pram outProb Address to save SVM prediction probability
 */
void iftSVMClassifyOVO_ProbabilitySingleSample(  iftSVM *svm, float *feats, int nFeats, int *outLabel, float *outProb);

/**
 *  @brief Find likelihood of each (label, bin) being forwarded to network. 
 *  @author Felipe Galvao
 *  @date Nov, 2019
 *
 *  @details As we can not know a priori the number of samples falling
 *    into each (label, bin) pair, the forwarding choice follows a random decision
 *    in which each (label, bin) pair has a probability weighted on the sample
 *    distribution and error rate over the training set.
 *           Current implementation might underestimate probabilities if the target
 *    fraction of samples is too high.
 *
 *  @param errorData Concatenated sample distribution and error rates (2*nBins x nLabels) matrix
 *  @param netFraction Overall percentage of samples to be forwarded to the network
 *  @return (nBins x nLabels) matrix with forwarding probabilities
 **/
iftMatrix * iftGetHybridBinProbabilities(iftMatrix *errorData, float netFraction);

/**
 *  @brief Decides if sample should be forwarded to network given svm prediction. 
 *  @author Felipe Galvao
 *  @date Nov, 2019
 *
 *  @param svmLabel Label predicted by SVM
 *  @param svmProb Probability assigned by p-SVM
 *  @param binProb Matrix with probability of all possible 
 *  @return True if sample should be classified by network
 **/
bool iftHybridDecision(int svmLabel, float svmProb, iftMatrix *binProb);

#endif
