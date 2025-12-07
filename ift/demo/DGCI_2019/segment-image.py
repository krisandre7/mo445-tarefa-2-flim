import pyift.pyift as ift
import argparse
import matplotlib.pyplot as plt


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

    parser = argparse.ArgumentParser(description="Segment image")
    parser.add_argument("-i", "--input-image", help="input image", required=True)
    parser.add_argument("-m", "--input-markers", help="input markers", required=True)
    parser.add_argument("-a", "--adjacency-radius", help="adjacency radius",
                        required=False, default=1.0, type=float)
    parser.add_argument("-d", "--use-distance", help="use local distance on dynamic ift",
                        action="count", dest="dist", required=False)
    parser.add_argument("-o", "--output-image", help="output segmented image", required=True)
    parser.add_argument("-s", "--segmentation-option", help="dynamic ift type", dest='segm', required=False,
                        choices=["root", "minroot", "object", "watershed", "watercut"], default='watercut')
    parser.add_argument("-p", "--marker-plato-height", help="height to enforce markers plato",
                        required=False, default=0, type=int, dest="plato")
    args = vars(parser.parse_args())

    img = ift.ReadImageByExt(args["input_image"])
    seeds = ift.ReadSeeds(img, args["input_markers"])
    A = ift.Circular(args['adjacency_radius'])
    mimg = ift.ImageToMImage(img, ift.LABNorm_CSPACE)
    label = None

    if args['segm'] == 'watercut':
        label = ift.WaterCut(mimg, A, seeds, None)

    elif args['segm'] == 'watershed':
        basins = ift.MImageBasins(mimg, A)
        label = ift.Watershed(basins, A, seeds, None)

    elif args['segm'] == 'object':
        if args['dist'] is None:
            label = ift.DynamicSetObjectPolicy(mimg, A, seeds, False)
        else:
            label= ift.DynamicSetObjectPolicy(mimg, A, seeds, True)

    elif args['segm'] == 'minroot':
        if args['dist'] is None:
            label = ift.DynamicSetMinRootPolicy(mimg, A, seeds, args['plato'], False)
        else:
            label = ift.DynamicSetMinRootPolicy(mimg, A, seeds, args['plato'], True)

    elif args['segm'] == 'root':
        if args['dist'] is None:
            label = ift.DynamicSetRootPolicy(mimg, A, seeds, args['plato'], False)
        else:
            label = ift.DynamicSetRootPolicy(mimg, A, seeds, args['plato'], True)

    ift.WriteImageByExt(label, args["output_image"])
    imageOverlay(img, label)


