import pyift.pyift as ift
import argparse


def unsupTrainOPF(Z, kmax):
#    mst = ift.CreateMST(Z)
#    graph = ift.MSTtoKnnGraph(mst, kmax)
    graph = ift.CreateKnnGraph(Z,kmax)
    ift.UnsupTrain(graph, ift.NormalizedCutPtr())
    return graph


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create KNN graph based on MST adjacency for clustering (unsupervised)")
    parser.add_argument("-d", "--dataset", help="input/output dataset", required=True)
    parser.add_argument("-k", "--kmax", help="number of neighbors (must be less than dataset number of samples)",
                        type=int, required=True)
    parser.add_argument("-g", "--knn-graph", help="output KNN graph", required=True)
    args = vars(parser.parse_args())

    Z = ift.ReadDataSet(args["dataset"])
    if Z.nsamples <= args["kmax"]:
        print("kmax (%d) must be less than number of dataset samples (%d)" % (args["kmax"], Z.nsamples))
        exit(-1)
    graph = unsupTrainOPF(Z, args["kmax"])
    ift.WriteKnnGraph(graph, args["knn_graph"])
    ift.WriteDataSet(Z, args["dataset"])
