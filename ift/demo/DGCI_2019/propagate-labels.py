import pyift.pyift as ift
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create complete graph for label propagation (semi-supervised)")
    parser.add_argument("-d", "--dataset", help="input/output dataset", required=True)
    parser.add_argument("-g", "--cpl-graph", help="output complete graph", required=True)
    args = vars(parser.parse_args())

    Z = ift.ReadDataSet(args["dataset"])
    graph = ift.SemiSupTrain(Z)
    ift.WriteDataSet(Z, args["dataset"])
    ift.WriteCplGraph(graph, args["cpl_graph"])
