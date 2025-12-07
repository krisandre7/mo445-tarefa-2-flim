import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 4:
    print "Usage: " + sys.argv[0] + " <data_path_prefix>_ <save_path_prefix> <dataset name>"
    exit()

# csv processing
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

classifierNumList = ["0", "1", "2"]
classifierNameList = ["OPFSup", "OPFSemi", "LogReg"]
datasetName = sys.argv[3]

for h in range(0, len(classifierNumList)):
    classifierStr = classifierNumList[h]
    classifierName = classifierNameList[h]
    
    fullRandStr = sys.argv[1] + "_100_" + classifierStr + ".csv"
    rootsRandStr = sys.argv[1] + "_120_" + classifierStr + ".csv"
    RWSStr = sys.argv[1] + "_200_" + classifierStr + ".csv"
    RDSStr = sys.argv[1] + "_300_" + classifierStr + ".csv"

    fullRandAcc, fullRandError, fullRandAnn = getAccArray(fullRandStr)
    rootsRandAcc, rootsRandError, rootsRandAnn = getAccArray(rootsRandStr)
    RWSAcc, RWSError, RWSAnn = getAccArray(RWSStr)
    RDSAcc, RDSError, RDSAnn = getAccArray(RDSStr)
    xVal = range(0,len(fullRandAcc))

    lines = plt.plot(xVal, fullRandAcc, xVal, rootsRandAcc, xVal, RWSAcc, xVal, RDSAcc)
#    plt.title("Active Learning using " + classifierName + " classifier on " + datasetName + " dataset")
    plt.ylabel('Accuracy over test set (%)')
    plt.xlabel('Active Learning Iterations')
    plt.setp(lines[0], label='Rand', alpha=0.8, c='g', ls='-', marker='+')
    plt.setp(lines[1], label='Rand-roots', alpha=0.8, c='b', ls='--', marker='1')
    plt.setp(lines[2], label='RWS', alpha=0.8, c='r', ls='-.', marker='2')
    plt.setp(lines[3], label='RDS', alpha=0.8, c='y', marker='3')
    plt.legend(loc=4)
    plt.savefig(sys.argv[2] + classifierName + "_" + datasetName + "_Acc.pdf", bbox_inches='tight');
    plt.savefig(sys.argv[2] + classifierName + "_" + datasetName + "_Acc.png", bbox_inches='tight');
    plt.clf()

#    lines = plt.plot(xVal, fullRandAnn, xVal, rootsRandAnn, xVal, RWSAnn, xVal, RDSAnn)
#    plt.title("Active Learning using " + classifierName + " classifier on " + datasetName + " dataset")
#    plt.ylabel('Annotated Samples (%)')
#    plt.xlabel('Active Learning Iterations')
#    plt.setp(lines[0], label='Rand', alpha=0.8, c='g', ls='-', marker='+')
#    plt.setp(lines[1], label='Rand-roots', alpha=0.8, c='b', ls='--', marker='1')
#    plt.setp(lines[2], label='RWS', alpha=0.8, c='r', ls='-.', marker='2')
#    plt.setp(lines[3], label='RDS', alpha=0.8, c='y', marker='3')
#    plt.legend(loc=1)
#    plt.savefig(sys.argv[2] + classifierName + "_" + datasetName + "_Ann.pdf", bbox_inches='tight');
#    plt.savefig(sys.argv[2] + classifierName + "_" + datasetName + "_Ann.png", bbox_inches='tight');
#    plt.clf()

exit()
