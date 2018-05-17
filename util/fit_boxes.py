import os
import json
from collections import namedtuple
import numpy as np
import cv2

LabelClass = namedtuple('LabelClass', ['name', 'color', 'id'])

# image should have R,G,B, channels, if json_path is provided, boxes are written to that path
def fit_boxes(img, classes, json_path=None):

    # get index (not id) of closest class for each pixel
    class_ind = np.power(img - np.stack([c.color for c in classes]).reshape(-1, 1, 1, 3), 2).sum(axis=3).argmin(axis=0)

    rimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    struct_elm = np.ones((2, 2), np.uint8)

    boxes = {c.name: [] for c in classes}
    for i, c in enumerate(classes):
        mask = (class_ind == i).astype(np.uint8)

        if mask.sum() < 20:
            continue

        # denoise mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct_elm)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct_elm)

        if mask.sum() < 20:
            continue

        # get mask contours
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for j, contour in enumerate(contours):

            # only use outer contours
            if hierarchy[0, j, 3] != -1:
                continue

            # get contour bounding box
            bbox = np.min(contour[:, :, 0]), np.max(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
            area = cv2.contourArea(contour)

            if  area < 20: # or area < (bbox[1]-bbox[0]) * (bbox[3] - bbox[2]) * 0.5:
                continue

            boxes[c.name].append(tuple(int(bbc) for bbc in bbox))

            # render box
            cv2.fillPoly(
                rimg,
                np.array([[[bbox[0], bbox[2]], [bbox[0], bbox[3]], [bbox[1], bbox[3]], [bbox[1], bbox[2]]]]),
                c.color)

    if json_path is not None:
        with open(json_path, 'w') as json_file:
            json.dump(boxes, json_file)

    return boxes, rimg