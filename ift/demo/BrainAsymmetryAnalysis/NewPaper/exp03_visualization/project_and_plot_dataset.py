import json
import os
import pdb
import sys

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import opf_dataset as opf

ITKSNAP = 'itksnap'
# ITKSNAP = 'ITK-SNAP'


if len(sys.argv) != 2:
    sys.exit("usage project_and_plot <opf_dataset.zip>")

print('- Reading Datasets')
dataset = opf.read_dataset(sys.argv[1]) # test control datasets

X = dataset['feats']

print('- Projecting by tSNE')
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2000)
X = tsne.fit_transform(X)

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)
ax.set_title('t-SNE projection')
line1,=ax.plot(X[:,0], X[:,1], 'ro', linewidth=0.1, picker=5, label='Controls Train')
plt.legend(handles=[line1])

# sets the axis limits
# xmin = X[:,0].min() - (abs(X[:,0].min()) * 2)
# xmax = X[:,0].max() + (abs(X[:,0].max()) * 2)
# ymin = X[:,1].min() - (abs(X[:,1].min()) * 2)
# ymax = X[:,1].max() + (abs(X[:,1].max()) * 2)
# print('old xmin, xmax: [%f, %f]' % (X[:,0].min(), X[:,0].max()))
# print('old ymin, ymax: [%f, %f]' % (X[:,1].min(), X[:,1].max()))
# print('xmin, xmax: [%f, %f]' % (xmin, xmax))
# print('ymin, ymax: [%f, %f]' % (ymin, ymax))
# plt.xlim(xmin, xmax)
# plt.ylim(ymin, ymax)


base_dir = '/home/xavecoder/workspace/exps/2018_MICCAI_Martins_novo'

def onpick1(event1):
    if (event1.artist != line1):
        return True
    N = len(event1.ind)
    if not N:
        return True
    for subplotnum, i in enumerate(event1.ind):
        pkey, orig_dir, patch_dir = dataset['ref_data'][i].split(";")
        print("\n{0}\n".format((pkey, orig_dir, patch_dir)))

        # command = '%s %s &' % (ITKSNAP, os.path.join(patch_dir, pkey + '_patch0.hdr'))
        # os.system(command)
        # command = '%s %s &' % (ITKSNAP, os.path.join(patch_dir, pkey + '_patch1.hdr'))
        # os.system(command)
        command = '%s %s &' % (ITKSNAP, os.path.join(orig_dir, pkey + '.hdr'))
        os.system(command)

fig.canvas.mpl_connect('pick_event', onpick1)

# fig.savefig('plots/controls_train_x_LTLE.png')

plt.show()
