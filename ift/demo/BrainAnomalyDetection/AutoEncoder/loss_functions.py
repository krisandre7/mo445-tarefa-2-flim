import keras.backend as K


# weights_tensor.shape = mask_tensor.shape = (batch_size, n_voxels)
def weighted_mse(weights_tensor):
    '''
    Custom loss function: Weighted Mean Square Error.

    Params
    ------
    weights_tensor: 2D Tensor(batch_size, n_voxels)
        Tensor with the weights for each voxel of each image in the batch size.
        In the simplest case, we have a sigle weight map replicated batch_size times.
    '''

    def loss(y_true, y_pred):
        '''
        Params
        ------
        y_true: 2D Tensor(batch_size, n_voxels)
            Tensor with the ground-truth (target) images from the batch.
            Each line of the tensor has a flatten target image with n_voxels voxels.

        y_pred: 2D Tensor(batch_size, n_voxels)
            Tensor with the predicted (reconstructed) images from the batch.
            Each line of the tensor has a flatten target image with n_voxels voxels.
        '''
        # y_true.shape = y_pred.shape = (batch_size, n_voxels)
        squared_error = K.square(y_true - y_pred)
        weighted_squared_error = squared_error * weights_tensor

        # for each image/row, sum its weighted errors
        weighted_mse_ = K.sum(weighted_squared_error, axis=1)

        return weighted_mse_

    return loss


# def masked_mse(mask_tensor, n_obj_voxels_tensor):
#     def loss(y_true, y_pred):
#         masked_squared_error = K.square(mask_tensor * (y_true - y_pred))
#         masked_mse_ = K.sum(masked_squared_error, axis=-1) / n_obj_voxels_tensor
#         return masked_mse_

#     return loss
