# @author Samuel Martins

# See an example of json plot config in "neg_mining/plot/Graph_configs.json"

import json
import math
import numpy as np
import os
import random
from scipy import stats
from scipy import interpolate
from sklearn.metrics import roc_curve
import sys
import pdb

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import setp


def print_error(msg, function):      
    sys.exit("\n*** PYTHON: ERROR in \"{0}\":\n{1}\n\n".format(function, msg))


def merge_results(files):
    """Merge results of a file set.

    Parameters
    ----------
    files: list of string
        List with the file paths.

    Returns
    -------
    X: list of float
        List with the x's values sorted in increasing order.

    Y: list of float
        List with the mean of the y's values for each x's.

    SEM: list of float
        List with the Standard Error of y for each x's.
        If there is an only y value, the sem is nan.

    Notes
    -----
    WARNING: The files must have the same x values and the same size.

    E.g: X = [0.5, 2], Y = [65.4, 98.3], SEM = [0.87, 1.2] means that the x_val=0.5 has the y_mean=65.4 with y_sem=0.87 
    """
    accs_dict = dict() # accs_dict[x_val] = [y_val1, y_val2, ...]
    
    for filename in files:
        filename = os.path.abspath(os.path.expanduser(filename))
        # print(filename)
        f = open(filename, "r")
        aux = f.read()
        aux = aux.split("\n")
        X = [float(d.split(" ")[0]) for d in aux if d]
        Y = [float(d.split(" ")[1]) for d in aux if d]
        XY = zip(X, Y)

        for x_val, y_val in XY:
            # print(x_val, y_val)
            if not x_val in accs_dict:
                accs_dict[x_val] = []
            accs_dict[x_val].append(y_val)
        f.close()
    
    scores = [] # list of tuples of the form (x_val, y_mean, sem)
    for x_val, y_vals in accs_dict.items():
        scores.append((x_val, np.mean(y_vals), stats.sem(y_vals)))

    scores = sorted(scores, key=lambda i: i[0]) # sort the list over the x's values
    
    X, Y, SEM = [], [], []
    for x_val, y_val, y_sem in scores:
        X.append(x_val)
        Y.append(y_val)
        SEM.append(y_sem)

    return np.array(X), np.array(Y), np.array(SEM)


def plot_bar(plot):
    graphs = plot["GRAPHS"]
    graph_keys = sorted(graphs.keys())

    for key in graph_keys:
    # for graph_name, graph in roc_graphs.items():
        graph_name = key
        graph = graphs[key]
        
        if graph["active"]:
            # if graph["figsize"] = null, it will be created a square
            plt.figure(**graph["figure"])
            plt.title(graph["title"]["label"], **graph["title"]["fontdict"])
            plt.grid(graph["grid"])

            plt.xlabel(graph["xlabel"]["label"], **graph["xlabel"]["fontdict"])
            plt.xlim(**graph["xlim"])
            plt.xscale(graph["xscale"])

            if graph["xticks"].get("ticks"):
                plt.xticks(graph["xticks"]["ticks"], graph["xticks"]["labels"], **graph["xticks"]["fontdict"])
            else:
                plt.xticks(np.linspace(graph["xlim"]["xmin"], graph["xlim"]["xmax"], graph["xticks"]["n_ticks"], endpoint=graph["xticks"]["endpoint"]), **graph["xticks"]["fontdict"])

            plt.ylabel(graph["ylabel"]["label"], **graph["ylabel"]["fontdict"])
            plt.ylim(**graph["ylim"])
            plt.yscale(graph["yscale"])

            if graph["yticks"].get("ticks"):
                plt.yticks(graph["yticks"]["ticks"], graph["yticks"]["labels"], **graph["yticks"]["fontdict"])
            else:
                plt.yticks(np.linspace(graph["ylim"]["ymin"], graph["ylim"]["ymax"], graph["yticks"]["n_ticks"], endpoint=graph["yticks"]["endpoint"]), **graph["yticks"]["fontdict"])


            curve_keys = sorted(graph["curves"].keys())
            
            for key in curve_keys:
                curve = graph["curves"][key]
                if curve["active"]:
                    print(key)
                    X, Y, SEM = merge_results(curve["files"])
                    plt.plot(X, Y, **curve["plot"])


            legend = plt.legend(loc=graph["legend"]["loc"], title=graph["legend"]["title"])
            setp(legend.get_title(), fontsize=graph["legend"]["title_fontsize"], family=graph["legend"]["family"])
            setp(legend.get_texts(), fontsize=graph["legend"]["text_fontsize"], family=graph["legend"]["family"])
            # pdb.set_trace()

            graph["savefig"]["fname"] = os.path.abspath(os.path.expanduser(graph["savefig"]["fname"]))
            print("- Saving plot in \"%s\"\n" % graph["savefig"]["fname"])
            if graph["savefig"]["format"].upper() == "PDF":
                pp = PdfPages(graph["savefig"]["fname"])
                plt.savefig(pp, **graph["savefig"])
                pp.close()
            else:
                plt.savefig(graph["savefig"]["fname"], **graph["savefig"])

            plt.cla()
            plt.clf()
            plt.close()



def main():
    if (len(sys.argv) == 1):
        print_error("plot_graph.py <plot_config1.json> <plot_config2.json> ...", "main")
    
    for i in range(1, len(sys.argv)):
        json_file = open(sys.argv[i])
    
        try:
            plot_config = json.load(json_file)
            #         print(json.dumps(configs, sort_keys=True, indent=4))
        except (ValueError, KeyError, TypeError):
            print_error("JSON format error", "main")

        plot_bar(plot_config)

if __name__ == "__main__":    
    sys.exit(main())