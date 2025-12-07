import os
import pdb
import sys

class Individual(object):
    def __init__(self, ind_name, pathnames):
        self.ind_name  = ind_name
        self.pathnames = pathnames

    def __repr__(self):
        print_msg  = ""
        print_msg += "%s\n" % self.ind_name
        print_msg += "Num of Imgs: %d\n" % len(self.pathnames)
        return print_msg

    def __str__(self):
        print_msg  = ""
        print_msg += "%s\n" % self.ind_name
        print_msg += "Num of Imgs: %d\n" % len(self.pathnames)
        for pathname in self.pathnames:
            print_msg += "%s\n" % pathname
        return print_msg


    @staticmethod
    def read_individuals(casia_dir):
        inds = []
        n_inds = len(os.listdir(casia_dir))

        for idx, ind_name in enumerate(os.listdir(casia_dir)):
            print("- Read Individual %d/%d" % (idx+1, n_inds))

            ind_dir = os.path.join(casia_dir, ind_name)
            pathnames = []
            for img_name in os.listdir(ind_dir):
                pathnames.append(os.path.join(ind_dir, img_name))
            inds.append(Individual(ind_name, pathnames))
        print("\n")
        return inds





def print_error(msg, function):
    sys.exit("ERROR in %s: %s" % (function, msg))


def validate_inputs(casia_dir, out_dir):
    if not os.path.isdir(casia_dir):
        msg = "Invalid CASIA dir: %s" % casia_dir
        print_error(msg, "validate_inputs")

    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            msg = "Out dir is a file and not a dir: %s" % out_dir
            print_error(msg, "validate_inputs")
    else:
        print("- Creating the output dir: %s" % out_dir)
        os.makedirs(out_dir)



def casia2ift(inds, out_dir):
    for ind_idx, individual in enumerate(inds):
        ift_label = "%06d" % (ind_idx+1)
        print("* Ind [%d/%d] - orig_label: %s --> ift_label: %s" % (ind_idx+1, len(inds), individual.ind_name, ift_label))

        for img_idx, orig_pathname in enumerate(individual.pathnames):
            filename     = int(orig_pathname.split("/")[-1].split(".")[0])
            filename     = ("%08d" % filename) + ".pgm"
            filename     = ift_label + "_" + filename
            out_pathname = os.path.join(out_dir, filename)
            
            print("\t- Img: [%d/%d]" % (img_idx+1, len(individual.pathnames)))
            os.system("convert %s -resize 200x200 %s" % (orig_pathname, out_pathname))
        




def main():
    if len(sys.argv) != 3:
        sys.exit("casia2ift <CASIA_dir> <out_dir>")

    casia_dir = os.path.abspath(os.path.expanduser(sys.argv[1]))
    out_dir   = os.path.abspath(os.path.expanduser(sys.argv[2]))

    validate_inputs(casia_dir, out_dir)

    inds = Individual.read_individuals(casia_dir)
    # sorts in ascending order according to the number images from each individual
    inds = sorted(inds, key=lambda ind: len(ind.pathnames))

    # converts and saves the images to IFT format
    casia2ift(inds, out_dir)

    

if __name__ == "__main__":
    sys.exit(main())

