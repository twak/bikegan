import numpy as np
import cv2

# metrics are:
# channels 1-4: distance to each side of the bounding box of the facade connected component (bottom, top, left, right), in multiples of the floor height
# channel 5: distance to facade boundary, in multiples of the floor height
# channel 6: constant floor height in fraction of the image height
def compute_metrics(mask, scale):

    metrics = np.zeros([6, mask.shape[0], mask.shape[1]], dtype=np.float32)

    ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # distance to bounding box sides
    # first component is background
    for i in range(1, ncomp):
        left = stats[i, 0]
        top = stats[i, 1]
        width = stats[i, 2]
        height = stats[i, 3]

        comp_mask = labels == i

        dbottom = np.tile(np.arange(height)[::-1].reshape(-1, 1), [1, width]) * comp_mask[top:top+height, left:left+width]
        dtop = np.tile(np.arange(height).reshape(-1, 1), [1, width]) * comp_mask[top:top+height, left:left+width]
        dleft = np.tile(np.arange(width), [height, 1]) * comp_mask[top:top+height, left:left+width]
        dright = np.tile(np.arange(width)[::-1], [height, 1]) * comp_mask[top:top+height, left:left+width]

        metrics[0, top:top+height, left:left+width] += dbottom
        metrics[1, top:top+height, left:left+width] += dtop
        metrics[2, top:top+height, left:left+width] += dleft
        metrics[3, top:top+height, left:left+width] += dright

    # distance to mask boundary
    # remove the image border, assume the mask stops at the image border
    mask_cropped = mask.copy()
    mask_cropped[[0, -1], :] = 0
    mask_cropped[:, [0, -1]] = 0
    metrics[4, :, :] = cv2.distanceTransform(mask_cropped, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_3)

    # multiply by scale
    metrics *= scale

    # constant unit length in fraction of the image height
    # (this is already adjusted for resizing of the image that might add black borders to top/bottom)
    metrics[5, :, :] = 1 / (scale * mask.shape[0])

    return metrics
