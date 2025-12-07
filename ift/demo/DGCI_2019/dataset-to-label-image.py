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

    parser = argparse.ArgumentParser(description="Display the original image with an overlay of the selected type")
    parser.add_argument("-i", "--input-image", help="input image", required=True)
    parser.add_argument("-d", "--input-dataset", help="input dataset", required=True)
    parser.add_argument("-s", "--superpixel-image", help="input superpixel image", required=True)
    parser.add_argument("-t", "--use-true-label", help="use true label data to label input image", action='count', dest='truelabel')
    parser.add_argument("-l", "--use-label", help="use label data to label input image", action='count', dest='label')
    parser.add_argument("-g", "--use-group", help="use group data to label input image", action='count', dest='group')
    parser.add_argument("-o", "--output-image", help="output labeled image", required=True)
    args = vars(parser.parse_args())

    label_type = None
    if args['truelabel'] is not None and args['label'] is None and args['group'] is None:
        label_type = ift.IFT_CLASS
    elif args['label'] is not None and args['truelabel'] is None and args['group'] is None:
        label_type = ift.IFT_LABEL
    elif args['group'] is not None and args['truelabel'] is None and args['label'] is None:
        label_type = ift.IFT_GROUP
    else:
        print("\nUser must select only one of the labeling options [-t (truelabel) -l (label) -g (group)]\n")
        exit(-1)

    Z = ift.ReadDataSet(args['input_dataset'])
    img = ift.ReadImageByExt(args['input_image'])
    suppxl = ift.ReadImageByExt(args['superpixel_image'])
    label = ift.DataSetToLabelImage(Z, suppxl, False, label_type)
    ift.WriteImageByExt(label, args['output_image'])
    imageOverlay(img, label)




