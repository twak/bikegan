import os
import json
from collections import namedtuple
import numpy as np
import cv2

LabelClass = namedtuple('LabelClass', ['name', 'color', 'id'])
LabelFit = namedtuple('LabelFit', ['max_count'])

# image should have R,G,B, channels and be of shape h x w x c
# if json_path is provided, boxes are written to that path
# boxes are stored as [xmin, xmax, ymin, ymax]
def fit_boxes(img, classes, fit_labels=None, json_path=None):

    if fit_labels is None:
        fit_labels = {c.name:-1 for c in classes}

    # get index (not id) of closest class for each pixel
    class_ind = np.power(img - np.stack([c.color for c in classes]).reshape(-1, 1, 1, 3), 2).sum(axis=3).argmin(axis=0)

    rimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    struct_elm = np.ones((2, 2), np.uint8)

    boxes = {cname: [] for cname, _ in fit_labels.items()}
    for i, c in enumerate(classes):
        if c.name not in fit_labels:
            continue

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

            # # render box
            # cv2.fillPoly(
            #     rimg,
            #     np.array([[[bbox[0], bbox[2]], [bbox[0], bbox[3]], [bbox[1], bbox[3]], [bbox[1], bbox[2]]]]),
            #     c.color)

    for class_name, class_boxes in boxes.items():
        # get class
        c = next((c for c in classes if c.name == class_name), None)

        # sort boxes by area and take n largest boxes (if requested)
        class_boxes = sorted(class_boxes, key=lambda box: (box[1]-box[0])*(box[3]-box[2]), reverse=True)
        if fit_labels[class_name].max_count >= 0:
            class_boxes = class_boxes[:fit_labels[class_name].max_count]
        boxes[c.name] = class_boxes

        for j, box in enumerate(class_boxes):

            # render box
            cv2.fillPoly(
                rimg,
                np.array([[[box[0], box[2]], [box[0], box[3]], [box[1], box[3]], [box[1], box[2]]]]),
                c.color)

    if json_path is not None:
        with open(json_path, 'w') as json_file:
            json.dump(boxes, json_file)

    return boxes, rimg

if __name__ == '__main__':
    fit_cls = [
        LabelClass('other', [0, 0, 0], 0),  # black borders or sky (id 0)
        LabelClass('background', [0, 0, 170], 1),  # background (id 1)
        LabelClass('facade', [0, 0, 255], 2),  # facade (id 2)
        LabelClass('molding', [255, 85, 0], 3),  # molding (id 3)
        LabelClass('cornice', [0, 255, 255], 4),  # cornice (id 4)
        LabelClass('pillar', [255, 0, 0], 5),  # pillar (id 5)
        LabelClass('window', [0, 85, 255], 6),  # window (id 6)
        LabelClass('door', [0, 170, 255], 7),  # door (id 7)
        LabelClass('sill', [85, 255, 170], 8),  # sill (id 8)
        LabelClass('blind', [255, 255, 0], 9),  # blind (id 9)
        LabelClass('balcony', [170, 255, 85], 10),  # balcony (id 10)
        LabelClass('shop', [170, 0, 0], 11),  # shop (id 11)
        LabelClass('deco', [255, 170, 0], 12),  # deco (id 12)
    ]

    fit_lbls = {'facade':LabelFit(-1), 'window':LabelFit(-1), 'door':LabelFit(-1), 'sill':LabelFit(-1), 'balcony':LabelFit(-1), 'shop':LabelFit(-1), 'molding':LabelFit(-1), 'cornice':LabelFit(-1)}

    img = cv2.imread('jon_b1110.png')[:, :, [2, 1, 0]]

    _, rimg = fit_boxes(img=img, classes=fit_cls, fit_labels=fit_lbls)

    cv2.imwrite('temp.png', rimg[:, :, [2, 1, 0]])
