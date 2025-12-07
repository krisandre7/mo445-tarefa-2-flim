import os
import sys
import csv

import numpy as np

from PIL import Image
import PIL

def GetVggInputShape():
    ySize = 200
    xSize = 200
    nBands = 3
    return (ySize, xSize, nBands)


def GetNClassesFromPathList(pathList):
  nClasses = 0
  for sample in range(0, len(pathList)):
    path = pathList[sample][0]
    base = os.path.basename(path)
    f = os.path.splitext(base)[0]
    label, ID = f.split("_")
    nClasses = max(nClasses, int(label))
  
  return nClasses

def GetNClassesFromCsv(csvPath):
  reader = csv.reader(open(csvPath, "r"), delimiter=',')
  samplePaths = list(reader)
  return GetNClassesFromPathList(samplePaths)

def LoadDatasetWithVggPreProc(csvPath):
  ySizeInput, xSizeInput, nBands = GetVggInputShape()

  # Read dataset csv
  reader = csv.reader(open(csvPath, "r"), delimiter=',')
  samplePaths = list(reader)

  # Build target tensors
  nSamples = len(samplePaths)
  nClasses = GetNClassesFromPathList(samplePaths)
  X = np.arange(nSamples * ySizeInput * xSizeInput * nBands, dtype=np.float32)
  X.resize(nSamples, ySizeInput, xSizeInput, nBands)
  Y = np.zeros(nSamples * nClasses)
  Y.resize(nSamples, nClasses)

  for sample in range(0, len(samplePaths)):
    path = samplePaths[sample][0]
    base = os.path.basename(path)
    f = os.path.splitext(base)[0]
    label, ID = f.split("_")
    
    img = Image.open(path.rstrip())
    if img.width != xSizeInput or img.height != ySizeInput:
        print("Expected %d x %d image but got a %d x %d one!" % (width, height, w, h))
        sys.exit(-1)
    if img.mode != "RGB":
        printf("Image was not opened as RGB!")
        sys.exit(-1)
    imgData = np.array(img)
    img.close()

    # RGB->BGR and subtract ImageNet mean (Initialization for Vgg16)
    X[sample,:,:,0] = imgData[:,:,2].copy() - 103.939
    X[sample,:,:,1] = imgData[:,:,1].copy() - 116.779
    X[sample,:,:,2] = imgData[:,:,0].copy() - 123.68
      
    Y[sample, int(label) - 1] = 1

  return X, Y


