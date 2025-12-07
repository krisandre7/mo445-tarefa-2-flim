import pyift.pyift as ift
import matplotlib.pyplot as plt
import argparse


def show(img, dpi=125):
    fig, ax=plt.subplots(dpi=dpi)
    ax.imshow(img)
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute dataset object saliency mapping")
    parser.add_argument("-d", "--input-dataset", help="input dataset", required=True)
    parser.add_argument("-g", "--cpl-graph", help="input complete graph", required=True)
    parser.add_argument("-s", "--superpixel-image", help="input superpixel image", required=True)
    parser.add_argument("-o", "--output-objmap", help="output saliency map", required=True)
    args = vars(parser.parse_args())

    Z = ift.ReadDataSet(args["input_dataset"])
    ift.SetStatus(Z, ift.IFT_TEST)
    graph = ift.ReadCplGraph(args["cpl_graph"])
    suppxl = ift.ReadImageByExt(args["superpixel_image"])
    ift.ClassifyWithCertaintyValues(graph, Z)
    objmap = ift.DataSetObjectMap(Z, suppxl, 255, 2)
    ift.WriteImageByExt(objmap, args["output_objmap"])

    show(objmap.ToPlot())
