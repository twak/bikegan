import os
import json
from collections import namedtuple
import numpy as np
import cv2

LabelClass = namedtuple('LabelClass', ['name', 'color', 'id'])
LabelFit = namedtuple('LabelFit', ['max_count'])

# image should have R,G,B, channels, if json_path is provided, circles are written to that path
# a circle is stored as (center x, center y, radius)
def fit_circles(img, classes, fit_labels=None, json_path=None):

    if fit_labels is None:
        fit_labels = {c.name:-1 for c in classes}

    # get index (not id) of closest class for each pixel
    class_ind = np.power(img - np.stack([c.color for c in classes]).reshape(-1, 1, 1, 3), 2).sum(axis=3).argmin(axis=0)

    rimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    # struct_elm = np.ones((2, 2), np.uint8)
    struct_elm = np.ones((6, 6), np.uint8)

    circles = {cname: [] for cname, _ in fit_labels.items()}
    for i, c in enumerate(classes):
        if c.name not in fit_labels:
            continue

        mask = (class_ind == i).astype(np.uint8)

        if mask.sum() < 10:
            continue

        # denoise mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct_elm)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct_elm)

        if mask.sum() < 10:
            continue

        # get mask contours
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # ncomp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for j, contour in enumerate(contours):

            # only use outer contours
            if hierarchy[0, j, 3] != -1:
                continue

            # get contour bounding box
            bbox = np.min(contour[:, :, 0]), np.max(contour[:, :, 0]), np.min(contour[:, :, 1]), np.max(contour[:, :, 1])
            center = ((bbox[0]+bbox[1]) / 2, (bbox[2]+bbox[3]) / 2)
            radius = np.sqrt(np.max((contour[:, :, 0] - center[0])**2 + (contour[:, :, 1] - center[1])**2))
            area = cv2.contourArea(contour)

            if area < 4:
                continue

            circles[c.name].append((int(center[0]), int(center[1]), radius))

    for class_name, class_circles in circles.items():
        # get class
        c = next((c for c in classes if c.name == class_name), None)

        # sort circles by radius and take n largest circles (if requested)
        class_circles = sorted(class_circles, key=lambda circle: circle[2], reverse=True)
        if fit_labels[class_name].max_count >= 0:
            class_circles = class_circles[:fit_labels[class_name].max_count]
        circles[c.name] = class_circles

        for j, circle in enumerate(class_circles):

            # render circle
            cv2.circle(
                img=rimg,
                center=circle[:2],
                radius=int(circle[2]),
                color=c.color,
                thickness=-1)

    if json_path is not None:
        with open(json_path, 'w') as json_file:
            json.dump(circles, json_file)

    return circles, rimg
