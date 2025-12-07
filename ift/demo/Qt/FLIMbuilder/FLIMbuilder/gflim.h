#ifndef GFLIM_H
#define GFLIM_H

#include "common.h"
#include "flim.h"


class GFLIM
{
public:
    GFLIM();
    ~GFLIM();

    /* Public member functions */
    bool IsArchitectureValid();

    bool TrainLayer(int layer);
    bool ExtractFeaturesFromLayer(int layer);
    bool ExtractFeaturesFromLayerForTraining(int layer);
    bool modelExists(int layer);

    QString ExtractFeaturesWithPython();
    void WriteSeedsFromImage(iftImage *markers_img, char* seeds_name, char *spx_name_file);
    //QString PrintGraph();

    void ExtractFeaturesFromGraphLayer(int layer);
    bool CreateGraphs(QString images_path);
    bool CreateGraph(QString images_path, QString basename);

    /* Variables */
    QJsonObject arch_json;
    QString images_dir;
    QString graphs_dir;
    QString param_dir;
    QString seeds_dir;
    QString object_dir;
    QString arch_path;
    QString labels_path;

    int num_nodes;
    int num_init_seeds;
    bool create_graphs;

    int device;

    iftFLIMArch *ift_arch;

};

#endif // GFLIM_H
