import numpy as np

def get_point(depthImg, viewMat, projMat):
    viewMat = np.array(viewMat).reshape((4, 4), order='F')
    projMat = np.array(projMat).reshape((4, 4), order='F')
    width = depthImg.shape[1]
    height = depthImg.shape[0]
    x = (2 * np.arange(0, width) - width) / width
    x = np.repeat(x[None, :], height, axis=0)
    y = -(2 * np.arange(0, height) - height) / height
    y = np.repeat(y[:, None], width, axis=1)
    z = 2 * depthImg - 1
    pix_pos = np.array([x.flatten(), y.flatten(), z.flatten(), np.ones_like(z.flatten())]).T
    tran_pix_world = np.linalg.inv(projMat @ viewMat)
    position = tran_pix_world @ pix_pos.T
    position = position.T
    position[:, :] /= position[:, 3:4]
    return position