import numpy
import os
import os.path
import binaryfile
import sys
import io
import zipfile
import deepdish
import scipy.sparse

def _get_label_file(filename):
    f = os.path.basename(filename)
    f = os.path.splitext(f)[0]
    return int(f[:f.find('_')])

def _get_id_file(filename):
    f = os.path.basename(filename)
    f = os.path.splitext(f)[0]
    try:
        return int(f[f.find('_') + 1:])
    except:
        print('Error get id from file:', filename)

def read_dataset(filename):
    zip = zipfile.ZipFile(filename)
    f = zip.open('info.data')

    nsamples = binaryfile.read_int(f)
    nfeats   = binaryfile.read_int(f)
    nclasses = binaryfile.read_int(f)
    nlabels  = binaryfile.read_int(f)
    function = binaryfile.read_int(f)

    id = numpy.zeros(nsamples, dtype=int)
    truelabel = numpy.zeros(nsamples, dtype=int)
    label = numpy.zeros(nsamples, dtype=int)
    weight = numpy.zeros(nsamples)
    status = numpy.zeros(nsamples, dtype=int)
    feats = numpy.zeros((nsamples, nfeats))

    for s in range(nsamples):
        id[s] = binaryfile.read_int(f)
        truelabel[s] = binaryfile.read_int(f)
        label[s] = binaryfile.read_int(f)

        weight[s] = binaryfile.read_float(f)
        status[s] = binaryfile.read_uchar(f)

        feats[s, :] = binaryfile.read_float_array2(f, nfeats)

    f.close()

    ref_data = None

    if 'ref_data.csv' in zip.namelist():
        f = zip.open('ref_data.csv')
        ref_data = numpy.loadtxt(f, dtype='str')



    return dict(id = id, feats = feats, label = label, truelabel = truelabel, status = status, weight = weight, function = function, ref_data = ref_data)

def save_dataset(filename, data):

    feats = data['feats']
    id = data['id']
    truelabel = data['truelabel'].flatten()
    label = data['label']
    weight = data['weight']
    status = data['status']

    nsamples, nfeats = feats.shape

    nclasses = len(numpy.unique(truelabel))
    nlabels = len(numpy.unique(label))

    function = data['function']


    zip = zipfile.ZipFile(filename, mode='w')
    f = io.BytesIO()

    binaryfile.write_int(f, nsamples)
    binaryfile.write_int(f, nfeats)
    binaryfile.write_int(f, nclasses)
    binaryfile.write_int(f, nlabels)
    binaryfile.write_int(f, function)

    for s in range(nsamples):
        binaryfile.write_int(f, id[s])
        binaryfile.write_int(f, truelabel[s])
        binaryfile.write_int(f, label[s])

        binaryfile.write_float(f, weight[s])
        binaryfile.write_uchar(f, status[s])
        # import pdb; pdb.set_trace()
        if scipy.sparse.issparse(feats[s, :]):
            feats_dense = feats[s,:].toarray()
        else:
            feats_dense = feats[s,:]
        binaryfile.write_float_array2(f, feats_dense.flatten())

    ref_data = data.get('ref_data', [])

    ref_data_type = 0

    if len(ref_data) > 0:
        ref_data_type = 4

    binaryfile.write_float_array2(f, numpy.ones(nfeats))#alpha
    binaryfile.write_int(f, ref_data_type)#ref data type

    zip.writestr('info.data', f.getvalue())

    header_info = "nfeats: 0\nhas_mean: 0\nhas_stdev: 0\nncomps: 0\nhas_w: 0\nhas_rotation_matrix: 0\nhas_whitening_matrix: 0"
    zip.writestr('feat_space.fsp', header_info)

    if  len(ref_data) > 0:
        zip.writestr('ref_data.csv', '\n'.join(ref_data.tolist()))

    zip.close()

def select_dataset(dataset, mask):
    new_data = {}
    
    new_data['function'] = dataset['function']
    new_data['weight']   = dataset['weight'][mask]
    new_data['status']   = dataset['status'][mask]
    new_data['label']    = dataset['label'][mask]
    new_data['truelabel']= dataset['truelabel'][mask]
    new_data['feats']    = dataset['feats'][mask, :]
    new_data['id']       = dataset['id'][mask]
    if dataset.get('ref_data', None)!=None:
        new_data['ref_data'] = dataset['ref_data'][mask]

    return new_data

def read_h5(filename):
    dataset = deepdish.io.load(filename)
    feats   = dataset['feats']

    nsamples, nfeats = feats.shape

    weight  = dataset.get('weight', numpy.zeros(nsamples))
    status  = dataset.get('status', numpy.zeros(nsamples))
    label   = dataset.get('label', numpy.zeros(nsamples))
    truelabel=dataset.get('truelabel', numpy.zeros(nsamples))
    function = dataset.get('function', 1)
    id = dataset.get('id', numpy.arange(nsamples))

    return dict(feats = feats, label = label, truelabel = truelabel, status = status, weight = weight, id = id, function = function)


def read_masters_h5(filename):
    dataset = deepdish.io.load(filename)
    feats   = dataset['X']
    truelabel=dataset['Y']

    if(truelabel.min()==0):
        truelabel += 1

    nsamples, nfeats = feats.shape

    weight  = numpy.zeros(nsamples)
    status  = numpy.zeros(nsamples)
    label   = numpy.zeros(nsamples)
    id = numpy.arange(nsamples)

    return dict(feats = feats, label = label, truelabel = truelabel, status = status, weight = weight, id = id, function = 1)


