#include <ift.h>

// Global TODO list
// -- make alternative 'backorder' selection after RDS/RWS guarantees cluster consistency
// -- add stop by reaching desired accuracy
// -- properly set verbose (-v) and relevant dialog
// -- active learning idea: try to classify some samples from each cluster
//      and based on root consistency, weight which cluster to pick samples
// -- optimize SemiSupTrain, single IFT instead of MST+IFT

iftDict *iftGetArguments(int argc, const char *argv[]);

#include "iftListAppend.h"
#include "iftRandomSelector.h"
#include "iftUnlabeledSelector.h"
#include "iftActiveClassifier.h"

int main(int argc, const char** argv) {
  size_t mem1 = iftMemoryUsed();

  // START argument parsing
  iftDict* args = iftGetArguments(argc, argv);

  char* ZtrainPath = iftGetStrValFromDict("--train-dataset", args);
  char* ZtestPath = iftGetStrValFromDict("--train-dataset", args);
  double kmax_perc = iftGetDblValFromDict("--kmax_perc", args);

  int activeMask = iftGetLongValFromDict("--active-method", args);
  // int activeUSAOpt = activeMask % 10; // Currently not used
  activeMask /= 10;
  iftSelectorFirstIterBehavior activeUSAFirstIter = activeMask % 10;
  activeMask /= 10;
  iftSelectorAlgorithm activeUSAlgorithm = activeMask % 10;

  int semisupMask;
  if (iftDictContainKey("semisup-method", args, NULL))
    semisupMask = iftGetLongValFromDict("semisup-method", args);
  else
    semisupMask = 100;
  // int semisupUSAOpt = semisupMask % 10; // Currently not used
  semisupMask /= 10;
  iftSelectorFirstIterBehavior semisupUSAFirstIter = semisupMask % 10;
  semisupMask /= 10;
  iftSelectorAlgorithm semisupUSAlgorithm = semisupMask % 10;
  semisupMask /= 10;

  iftClassifierAlgorithm classifierAlgo = iftGetLongValFromDict("--classifier", args);
  int maxActiveIters;
  if (iftDictContainKey("--active-iters", args, NULL))
    maxActiveIters = iftGetLongValFromDict("--active-iters", args);
  else
    maxActiveIters = 50;
  int verbose;
  if (iftDictContainKey("--verbose", args, NULL))
    verbose = iftGetLongValFromDict("--verbose", args);
  else
    verbose = 0;
  iftCSV * resultsCSV = NULL;
  char * pathCSV = NULL;
  // Format: iteration(int),accuracy(%),annotated samples(%),time(ms),missing classes(int)
  if (iftDictContainKey("--write-csv", args, NULL)) {
    pathCSV = iftGetStrValFromDict("--write-csv", args);
    resultsCSV = iftCreateCSV(maxActiveIters + 1, 5);
    strcpy(resultsCSV->data[0][0], "Iteration");
    strcpy(resultsCSV->data[0][1], "Accuracy");
    strcpy(resultsCSV->data[0][2], "Annotated samples");
    strcpy(resultsCSV->data[0][3], "Execution time");
    strcpy(resultsCSV->data[0][4], "Missing classes");
  }

  iftDestroyDict(&args);
  // END argument parsing

  // Initialize
  iftRandomSeed(time(NULL));
//  iftRandomSeed(0);
  iftDataSet *Z = iftReadOPFDataSet(ZtrainPath);
  iftDataSet *Ztest = iftReadOPFDataSet(ZtestPath);
  if (verbose)
    printf("Using %d samples as unlabeled.\n", Z->nsamples);

  // Result collection
  int nShown = 0; // samples seen by oracle
  int nCorrected = 0; // samples corrected by oracle 
  int iterCount = 0;
  double accuracy = 0.0;
  int *seenClasses = iftAllocIntArray(Z->nclasses);
  int missingClasses = Z->nclasses;


  // -- Data organization phase
  timer *t1_org = iftTic();
  // UnsupOPF Clustering
  int kmax  = (int) iftMax(kmax_perc * Z->nsamples, 1);
  iftKnnGraph *graph = iftCreateKnnGraph(Z, kmax);
  iftUnsupTrain(graph, iftNormalizedCut);
  // Store cluster label data because it will be overriden during active learning
  int nClusters = Z->nlabels;
  if (verbose) {
    printf("Obtained %d clusters.\n", nClusters);
    printf("%d nodes in the clustered graph (k=%d kmax=%d).\n", 
        graph->nnodes, graph->k, graph->kmax);    
  }

  // Structure for choosing active learning samples
  iftUnlabeledSelector * activeSelector = iftCreateUnlabeledSelector(
      activeUSAlgorithm, activeUSAFirstIter, Z, graph);

  // Structure for choosing semisup classifier unlabeled samples
  iftUnlabeledSelector * semisupSelector = NULL;
  if (classifierAlgo == OPFSEMI_CLASSIFIER) {
    semisupSelector = iftCreateUnlabeledSelector(
        semisupUSAlgorithm, semisupUSAFirstIter, Z, graph);
  }

  // Prepare classifier
  iftActiveClassifier * activeClassifier = 
    iftCreateActiveClassifier(classifierAlgo, Z, semisupSelector);

  timer *t2_org = iftToc();
  if (verbose)
    iftPrintCompTime(t1_org, t2_org, "Data organization"); 

  // Active learning will use both label and status fields from samples
  iftResetDataSet(Z);
  iftSetStatus(Z, IFT_UNKNOWN);

  // -- Active Learning Loop
  timer *t1_act = iftTic();
  for (int activeIter = 1; activeIter <= maxActiveIters; ++activeIter) {
    timer *t1_actIter = iftTic();
    int iterShown = 0;
    int iterCorrected = 0;

    // Reset SemiSup unlabeled samples
    for (int s = 0; s < Z->nsamples; ++s) {
      if (Z->sample[s].status != IFT_TRAIN)
        Z->sample[s].label = 0;
    }

    // -- Sample selection
    int numSelected = (activeIter == 1 ? nClusters : 2 * Z->nclasses);
    iftList * selectedSamples = NULL;

    selectedSamples = iftPickFromUnlabeledSelector(activeSelector, numSelected, activeClassifier);

    if (selectedSamples->n > 0) {
      // Process selected samples
      iftNode * listIter = selectedSamples->head;
      while (listIter != NULL) {
        iterShown++;
        int sample = iftRemoveListNode(selectedSamples, &listIter, false);

        if (Z->sample[sample].label == 0 && activeClassifier->isTrained)
          iftActiveClassifierClassifySample(activeClassifier, Z, sample);

        if (Z->sample[sample].label != Z->sample[sample].truelabel) {
          Z->sample[sample].label = Z->sample[sample].truelabel;
          iterCorrected++;
        }
        Z->sample[sample].status = IFT_TRAIN;

        // Update how many classes yet to find
        if (missingClasses > 0) {
          if (seenClasses[Z->sample[sample].truelabel - 1] == 0) {
            seenClasses[Z->sample[sample].truelabel - 1]++;
            missingClasses--;
          }
        }
      }
      iftDestroyList(&selectedSamples);

      // Feedback
      if (verbose) {
        if (missingClasses)
          printf("Missing %d out of %d classes in iteration %d.\n", 
              missingClasses, Z->nclasses, activeIter);
        printf("%d samples shown to oracle of which %d needed correction.\n",
            iterShown, iterCorrected);
      }

      // -- Train Classifier
      iftTrainActiveClassifier(activeClassifier);

      // Classify Test
      iftResetDataSet(Ztest);
      iftActiveClassifierClassifyDataSet(activeClassifier, Ztest);

      // Collect data
      timer *t2_actIter = iftToc();
      iterCount++;
      accuracy = iftTruePositives(Ztest);
      nShown += iterShown;
      nCorrected += iterCorrected;
      if (verbose)
        printf("Accuracy in iter %d = %.2f%%\n", activeIter, accuracy * 100);
      if (resultsCSV) {
        sprintf(resultsCSV->data[activeIter][0], "%d", activeIter);
        sprintf(resultsCSV->data[activeIter][1], "%.5f", accuracy * 100);
        sprintf(resultsCSV->data[activeIter][2], "%.5f", (double)iterCorrected/(double)iterShown);
        sprintf(resultsCSV->data[activeIter][3], "%.5f", iftCompTime(t1_actIter, t2_actIter));
        sprintf(resultsCSV->data[activeIter][4], "%d", missingClasses);
      }
    } else {
      // TODO centralize this destruction
      iftDestroyList(&selectedSamples);
      break;
    }
  }
  timer *t2_act = iftToc();
  if (verbose) {
    iftPrintCompTime(t1_act, t2_act, "Total active learning time"); 
    printf("Finished in %d active learning cycles and final accuracy %.2f%%\n\n",
      iterCount, accuracy * 100);
  }
  if (resultsCSV)
    iftWriteCSV(resultsCSV, pathCSV, ',');

  // Clean up
  iftDestroyUnlabeledSelector(&activeSelector);
  iftDestroyActiveClassifier(&activeClassifier);
  iftDestroyDataSet(&Z);
  iftDestroyDataSet(&Ztest);
  free(seenClasses);
  iftDestroyKnnGraph(&graph);
  iftDestroyCSV(&resultsCSV);
  free(ZtrainPath);
  free(ZtestPath);

  size_t mem2 = iftMemoryUsed();

  if (verbose)
    iftVerifyMemory(mem1, mem2);

  return 0;
}

/* sources */
#define ACTIVE_METHOD_HELP "Active Learning algorithm from US.\n\
  Unlabeled selector (US) is defined by a 3 digit number xyz:\n\
 x=Main Selector Algorithm | y=First Iteration Behavior | z=Other options\n\
 0 = ALL_SAMPLES           | 0 = FIRST_ITER_DEFAULT     | WIP\n\
 1 = FULL_RANDOM           | 1 = FIRST_ITER_RANDOM      |\n\
 2 = ACTIVE_RWS            | 2 = FIRST_ITER_ROOTS       |\n\
 3 = ACTIVE_RDS            |                            |\n"

#define CLASSIFIER_HELP "Which classifier is used:\n\
  0 = OPFSup\n\
  1 = OPFSemi\n\
  2 = LogReg (WIP)"

iftDict *iftGetArguments(int argc, const char *argv[]) {
  iftCmdLineOpt cmd_line_opts[] = {
    {.short_name = "-d", .long_name = "--train-dataset", .has_arg=true,
      .arg_type=IFT_STR_TYPE, .required=true, .help="Dataset to run active learning."},
    {.short_name = "-e", .long_name = "--test-dataset", .has_arg=true,
      .arg_type=IFT_STR_TYPE, .required=true, .help="Dataset to test against for accuracy."},
    {.short_name = "-k", .long_name = "--kmax_perc", .has_arg=true,
      .arg_type=IFT_DBL_TYPE, .required=true, .help="kmax for OPF based on % of samples."},
    {.short_name = "-a", .long_name = "--active-method", .has_arg=true,
      .arg_type=IFT_LONG_TYPE, .required=true, .help=ACTIVE_METHOD_HELP},
    {.short_name = "-s", .long_name = "--semisup-method", .has_arg=true,
      .arg_type=IFT_LONG_TYPE, .required=false, .help="See --active-method."},
    {.short_name = "-c", .long_name = "--classifier", .has_arg=true,
      .arg_type=IFT_LONG_TYPE, .required=true, .help=CLASSIFIER_HELP},
    {.short_name = "-m", .long_name = "--active-iters", .has_arg=true,
      .arg_type=IFT_LONG_TYPE, .required=false, .help="Max active learning iterations."},
    {.short_name = "-w", .long_name = "--write-csv", .has_arg=true,
      .arg_type=IFT_STR_TYPE, .required=false, .help="Saves results in <path.csv>."}
  };
  int n_opts = sizeof(cmd_line_opts) / sizeof (iftCmdLineOpt);

  char program_description[IFT_STR_DEFAULT_SIZE] = 
    "This program runs active learning methods based on saito's methodology.";

  // Parser Setup
  iftCmdLineParser *parser =
    iftCreateCmdLineParser(program_description, n_opts, cmd_line_opts);
  iftDict *args = iftParseCmdLine(argc, argv, parser);
  iftDestroyCmdLineParser(&parser);

  return args;
}

