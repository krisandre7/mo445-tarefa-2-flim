import pyift.pyift as ift
import matplotlib.pyplot as plt
import argparse
import sys
import os

def show(img, dpi=125):
    fig, ax=plt.subplots(dpi=dpi)
    ax.imshow(img)
    plt.show()


def imageOverlay(original, label):
    A = ift.Circular(1.0)
    tmp = ift.CopyImage(original)
    cmap = ift.CreateColorTable(ift.MaximumValue(label) + 1)
    ift.DrawLabels(tmp, label, cmap, A, True, 0.2)
    show(tmp.ToPlot())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Segment image using ISF Mix Mean")
    parser.add_argument("-i", "--input-image", help="input image", required=True)
    parser.add_argument("-n", "--number-superpixels", help="number of superpixels", required=True, type=int)
    parser.add_argument("-o", "--output-image", help="output image of superpixels labels", required=True)
    args = vars(parser.parse_args())

#    img = ift.ReadImageByExt(args["input_image"])
#    suppxl = ift.Extract_ISF_MIX_ROOT_Superpixels(img, args["number_superpixels"], 0.5, 12, 10, 2)
#    ift.WriteImageByExt(suppxl, args["output_image"])
#    imageOverlay(img, suppxl)

    n = args["number_superpixels"]
    os.system("iftRISF_segmentation -i " + args["input_image"] + " -o temp.pgm -n " + str(3 * n) + " -s 1")
    os.system("iftRISF_segmentation -i " + args["input_image"] + " -o " + args["output_image"] + " -n " + str(n) + " -s 4" + " -l temp.pgm")
    img    = ift.ReadImageByExt(args["input_image"])
    suppxl = ift.ReadImageByExt(args["output_image"])
    aux    = ift.SmoothRegionsByDiffusion(suppxl,img,0.5,2)
    A      = ift.Circular(1.5)
    suppxl = ift.RelabelRegions(aux,A)
    ift.WriteImageByExt(suppxl,args["output_image"])    
    imageOverlay(img, suppxl)
    
#    suppxl = ift.SelectAndPropagateRegionsAboveAreaByColor(img, aux, 50)
