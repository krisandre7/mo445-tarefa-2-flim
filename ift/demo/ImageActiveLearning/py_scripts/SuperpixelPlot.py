import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getPlotData(csvPath, xCol, yCol):
    reader = csv.reader(open(csvPath, "rb"), delimiter=',')
    data = list(reader)
    xData = []
    yData = []
    for i in range(1,len(data)):
        xData += [data[i][xCol]]
        yData += [data[i][yCol]]
    return xData, yData 

if len(sys.argv) < 12:
    print "Usage: %s <output.pdf> <xName> <xMin> <xMax> <xStep> <yName> <yMin> <yMax> <yStep> <nMethods> (<csv_path> <name> <color> <lineStyle> <marker> <xColumn> <yColumn>)*nMethods" % sys.argv[0]
    sys.exit(-1)

FIXED_ARG_N=11
METHOD_ARG_N=7
target = sys.argv[1]
xName = sys.argv[2]
xMin = float(sys.argv[3])
xMax = float(sys.argv[4])
xStep = float(sys.argv[5])
yName = sys.argv[6]
yMin = float(sys.argv[7])
yMax = float(sys.argv[8])
yStep = float(sys.argv[9])
nMethods = int(sys.argv[10])

if len(sys.argv) < FIXED_ARG_N + METHOD_ARG_N * abs(nMethods):
    print "Missing method parameters. Provide params:"
    print "target = %s" % sys.argv[1]
    print "xName = %s" % sys.argv[2]
    print "xMin = %s" % sys.argv[3]
    print "xMaxx = %s" % sys.argv[4]
    print "xStep = %s" % sys.argv[5]
    print "yName = %s" % sys.argv[6]
    print "yMin = %s" % sys.argv[7]
    print "yMax = %s" % sys.argv[8]
    print "yStep = %s" % sys.argv[9]
    print "nMethods = %s" % sys.argv[10]
    for i in range(11, len(sys.argv)):
        if (i - FIXED_ARG_N) % METHOD_ARG_N == 0:
            print "Method %d (csv, name, color, linestyle, xCol, yCol:" % (((i - FIXED_ARG_N) / METHOD_ARG_N) + 1)
        print "param = %s" % sys.argv[i]
    sys.exit(-1)

lines = []
labels = []
colors = []
#colorTb = ['g', 'orange', 'b', 'r', 'olive', 'm', 'c', 'y', 'teal', 'orangered']
#colorTb = ['blue', 'green', 'red', 'darkred', 'violet', 'orange', 'teal', 'purple']
colorTb = ['deepskyblue', 'blue','green', 'red', 'darkred', 'violet', 'orange', 'teal', 'purple']
lineStyles = []
#lineStyleTb = ['solid', 'dashed', 'dashdot', 'dotted']
#lineStyleTb = ['solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashdot', 'dashdot', 'dashdot']
lineStyleTb = ['solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashdot', 'dashdot', 'dashdot']
markers = []
markersTb = ['^', 'D', 's', 'p', 'o', '', '.']
for methodIdx in range(nMethods):
    argOffset = FIXED_ARG_N + METHOD_ARG_N * methodIdx
    path = sys.argv[argOffset]
    label = sys.argv[argOffset + 1]
    color = int(sys.argv[argOffset + 2])
    lineStyle = int(sys.argv[argOffset + 3])
    marker = int(sys.argv[argOffset + 4])
    xCol = int(sys.argv[argOffset + 5])
    yCol = int(sys.argv[argOffset + 6])
    xData, yData = getPlotData(path, xCol, yCol)
    lines += plt.plot(xData, yData, alpha=0.1)
    labels += [label]
    colors += [colorTb[color]]
    lineStyles += [lineStyleTb[lineStyle]]
    markers += [markersTb[marker]]

if xMax > xMin and yMax > yMin:
    plt.axis([xMin,xMax,yMin,yMax])
    if xStep > 0:
        plt.xticks(np.arange(xMin, xMax, xStep))
    if yStep > 0:
        plt.yticks(np.arange(yMin, yMax, yStep))
else:
    print "WARNING: Invalid axis intervals, using default values."

plt.xlabel(xName)
plt.ylabel(yName)
for i in range(0,len(lines)):
    plt.setp(lines[i], label=labels[i], alpha=0.8, c=colors[i], ls='-', linestyle=lineStyles[i], marker=markers[i], linewidth=2.0, mew=0.0)

plt.legend(loc=0)
plt.savefig(target, bbox_inches='tight');

