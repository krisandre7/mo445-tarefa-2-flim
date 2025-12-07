import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

#Fixed Classifier and changing active algo
#0 = OPFSup
#1 = OPFSemi
#2 = LogReg
classifierStr = "0"
classifierName = "OPFSup"
activeStr = "_300_"
activeName = "RDS"
datasetName = "Larvas"

def getAccArray(str):
    reader = csv.reader(open(str, "rb"), delimiter=',')
    x = list(reader)
    y = []
    error = []
    annotation = []
    for i in range(1,len(x)):
        y += [x[i][1]]
        error += [x[i][2]]
        annotation += [x[i][3]]
    return y, error, annotation

OPFSupStr = sys.argv[1] + activeStr + "0.csv"
OPFSemiStr = sys.argv[1] + activeStr + "1.csv"
LogRegStr = sys.argv[1] + activeStr + "2.csv"

OPFSupAcc, OPFSupError, OPFSupAnn = getAccArray(OPFSupStr)
OPFSemiAcc, OPFSemiError, OPFSemiAnn = getAccArray(OPFSemiStr)
LGAcc, LGError, LGAnn = getAccArray(LogRegStr)
xVal = range(0,len(OPFSupAcc))

lines = plt.plot(xVal, OPFSupAcc, xVal, OPFSemiAcc, xVal, LGAcc)
plt.title("Active Learning using " + activeName + " on " + datasetName + " dataset")
plt.ylabel('Accuracy over test set (%)')
plt.xlabel('Active Learning Iterations')
plt.setp(lines[0], label='OPFSup', alpha=0.8, c='g', ls='-', marker='+')
plt.setp(lines[1], label='OPFSemi', alpha=0.8, c='b', ls='--', marker='1')
plt.setp(lines[2], label='LogReg', alpha=0.8, c='r', ls='-.', marker='2')
plt.legend(loc=3)
#plt.savefig(sys.argv[2] + classifierName + "_" + datasetName + "_Acc.pdf", bbox_inches='tight');
#plt.savefig(sys.argv[2] + classifierName + "_" + datasetName + "_Acc.png", bbox_inches='tight');
#plt.clf()
plt.show()
