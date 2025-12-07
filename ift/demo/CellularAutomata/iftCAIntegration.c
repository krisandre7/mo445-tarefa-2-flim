#include "ift.h"
#include "include/ca.h"

#define MAX_ITERATIONS 3  // TC parameter from paper
#define LAMBDA 0.05f      // Λ parameter from paper
#define CA_INTEGRATION_DEBUG 1

iftCASaliencyModel *iftCreateCASaliencyModel(int n_layers) {
  iftCASaliencyModel *ca = calloc(1, sizeof(iftCASaliencyModel));
  ca->n_prob_maps = n_layers;
  ca->prob_maps = calloc(n_layers, sizeof(iftFImage *));
  ca->thresholds = calloc(n_layers, sizeof(float));

  return ca;
}

void iftDestroyCASaliencyModel(iftCASaliencyModel **ca_model) {
  if ((*ca_model) != NULL) {
    if ((*ca_model)->prob_maps != NULL) {
      for (size_t n=0; n < (*ca_model)->n_prob_maps; n++) {
        if ((*ca_model)->prob_maps[n] != NULL) {
          iftDestroyFImage(&((*ca_model)->prob_maps[n]));
        }
      }
      iftFree((*ca_model)->prob_maps);
      iftFree((*ca_model)->thresholds);
    }
    iftFree((*ca_model));
  }
}

float iftComputeCASigma(iftCASaliencyModel *ca_model, int p, int m, int N) {
  float sigma = 0.0;
  iftAdjRel *adj_rel = iftRectangular(N, N);
  iftVoxel u = iftFGetVoxelCoord(ca_model->prob_maps[0], p);

  for (size_t adj = 0; adj < adj_rel->n; adj++) {
    for (size_t k = 0; k < ca_model->n_prob_maps; k++) {
      if (k == m && adj == 0) {
        continue;
      }
      
      iftVoxel v = iftGetAdjacentVoxel(adj_rel, u, adj);
      if (iftValidVoxel(ca_model->prob_maps[k], v)) {
        int q = iftFGetVoxelIndex(ca_model->prob_maps[k], v);
        float value = ca_model->prob_maps[m]->val[q];
        
        int binary_state = value >= ca_model->thresholds[k] ? 1 : -1;
        sigma += binary_state;
      }
    }
  }

  iftDestroyAdjRel(&adj_rel);

  return sigma;
}

float iftCAToLogOdds(float value) {
  if (value <= 0) {
    value = 0.0001f;
  }
  else if (value >= 1) {
    value = 0.9999f;
  }

  return log(value / (1.0f - value));
}

float iftCAToProb(float value) {
  float exp_value = exp(value);

  return exp_value / (1.0f + exp_value);
}

void iftUpdateSaliencyCA(iftCASaliencyModel *ca_model) {
  iftFImage **new_prob_maps = calloc(ca_model->n_prob_maps, sizeof(iftFImage *));
  #pragma omp parallel for
  for (size_t m = 0; m < ca_model->n_prob_maps; m++) {
      new_prob_maps[m] = iftCreateFImageFromFImage(ca_model->prob_maps[m]);
  }

  size_t n = ca_model->prob_maps[0]->n;
  #pragma omp parallel for collapse(2)
  for (size_t m = 0; m < ca_model->n_prob_maps; m++) {
    for (size_t p = 0; p < n; p++) {
        float sigma = iftComputeCASigma(ca_model, p, m, 3);

        // Calculate new values using update rule
        float current_log_odds = iftCAToLogOdds(
          ca_model->prob_maps[m]->val[p]
        );
        float new_log_odds = current_log_odds + sigma * LAMBDA;
        new_prob_maps[m]->val[p] = iftCAToProb(new_log_odds);
    }
  }

  for (size_t m = 0; m < ca_model->n_prob_maps; m++) {
    iftDestroyFImage(&(ca_model->prob_maps[m]));
  }
  iftFree(ca_model->prob_maps);
  ca_model->prob_maps = new_prob_maps;
}

iftImage *iftIntegrateSaliencyMaps(
  char *output_dir, char *img_basename, iftCASaliencyModel *ca_model
) {
  
  for (int it = 0; it < MAX_ITERATIONS; it++) {
    iftUpdateSaliencyCA(ca_model);
    if (CA_INTEGRATION_DEBUG) {
      char output_path[1024];
      for (size_t m = 0; m < ca_model->n_prob_maps; m++) {
        iftImage *evolved_prob_map = iftCreateImage(
          ca_model->prob_maps[m]->xsize, ca_model->prob_maps[m]->ysize,
          ca_model->prob_maps[m]->zsize
        );
        #pragma omp parallel for
        for (size_t p = 0; p < ca_model->prob_maps[m]->n; p++) {
          evolved_prob_map->val[p] = (int)(ca_model->prob_maps[m]->val[p] * 255);
        }
        sprintf(
          output_path, "%s/%s_evolved_prob_map_it%d_layer%ld.png", output_dir, img_basename, it, m + 1
        );
        iftWriteImageByExt(evolved_prob_map, output_path);
        iftDestroyImage(&evolved_prob_map);
      }
    }
  }

  iftImage *integrated_sal = iftCreateImage(
    ca_model->prob_maps[0]->xsize,
    ca_model->prob_maps[0]->ysize,
    ca_model->prob_maps[0]->zsize
  );

  #pragma omp parallel for
  for (size_t p = 0; p < ca_model->prob_maps[0]->n; p++) {
    float sum = 0.0f;
    for (size_t m = 0; m < ca_model->n_prob_maps; m++) {
      sum += ca_model->prob_maps[m]->val[p];
    }
    integrated_sal->val[p] = (int)( (sum / ca_model->n_prob_maps) * 255.0f );
  }

  return integrated_sal;
}

int main(int argc, char **argv) {
  timer *tstart;
  int memory_start, memory_end;
  memory_start = iftMemoryUsed();
  tstart = iftTic();

  if (argc != 4) {
    iftError(
      "Usage: iftCA <P1> <P2> <P3>\n"
      "P1: CA Output Folder to combine saliencies.\n"
      "P2: Path to save combined saliencies.\n"
      "P3: N Layers.\n", 
      "main"
    );
  }

  char input_folder[512];
  char input_saliency_path[1024];
  char output_path[1024];
  int n_layers = atoi(argv[3]);

  sprintf(input_folder, "%s/layer_1/", argv[1]);
  iftFileSet *fs = iftLoadFileSetFromDirBySuffix(input_folder, "_prob_map.png", 1);
  
  for (size_t i=0; i < fs->n; i++) {
    char *img_basename = iftFilename(fs->files[i]->path, "_prob_map.png");
    iftCASaliencyModel *ca = iftCreateCASaliencyModel(n_layers);

    printf("\nRunning for image: %s\n", img_basename);
    for (size_t n=0; n < n_layers; n++) {
      sprintf(input_saliency_path, "%s/layer_%ld/%s_prob_map.png", argv[1], n + 1, img_basename);
      iftImage *saliency = iftReadImageByExt(input_saliency_path);
      ca->prob_maps[n] = iftCreateFImage(saliency->xsize, saliency->ysize, saliency->zsize);
      ca->thresholds[n] = (float)iftOtsu(saliency) / 255;
      if (ca->thresholds[n] == 0) {
        ca->thresholds[n] = 1;
      }
      if (CA_INTEGRATION_DEBUG) {
        sprintf(
          output_path, "%s/%s_prob_map_layer%ld.png", argv[2], img_basename, n
        );
        iftWriteImageByExt(saliency, output_path);
      }

      #pragma omp parallel for
      for (size_t p = 0; p < saliency->n; p++) {
        ca->prob_maps[n]->val[p] = (float)saliency->val[p] / 255;
      }
      iftDestroyImage(&saliency);
    }

    iftImage *integrated_saliency = iftIntegrateSaliencyMaps(
      argv[2], img_basename, ca
    );

    sprintf(output_path, "%s/%s.png", argv[2], img_basename);
    iftWriteImageByExt(integrated_saliency, output_path);

    iftDestroyImage(&integrated_saliency);
    iftFree(img_basename);
    iftDestroyCASaliencyModel(&ca);
  }

  puts("\nDone ....");
  iftDestroyFileSet(&fs);
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));
  memory_end = iftMemoryUsed();
  iftVerifyMemory(memory_start, memory_end);

  return 0;
}