import tensorflow as tf
import numpy as np
import os
import anchor
import cv2
import matplotlib.pyplot as plt
from utils import draw_boxes
import config

C = config.CLASS
print(C)
config.BOX = 4

anchors, _, _ = anchor.kmeans(anchor.wh, k=config.BOX, dist=anchor.dist, seed=2)  # get the final anchor boxes
# the widths and heights are ranging between 0-1

anchors[::2] = anchors[::2] * config.GRID_H
anchors[1::2] = anchors[1::2] * config.GRID_W  # bringing anchor boxes to grid scale

print("The final anchor boxes are : ", anchors)

train_image_paths = []
enc_obj = []
train_labels = np.zeros((2500, config.GRID_H, config.GRID_W, config.BOX, (5 + C)))


def image_preprocess(image_path, label):

    image_path = image_path.numpy().decode("utf-8")
    # image = tf.io.read_file(image_path)
    # image = tf.io.decode_image(image, channels=3, dtype='float32')

    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(config.IMAGE_H, config.IMAGE_W))
    image = np.asarray(image, dtype="float32")
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if image is None: print("Cannot find Image!")
    # image = cv2.resize(image, (config.IMAGE_H, config.IMAGE_W))
    # image = tf.image.resize(image, (config.IMAGE_H, config.IMAGE_W))  # resize to (416, 416)
    image = image / 255.0  # Normalize
    label = tf.cast(label, dtype=tf.float32)

    return image, label


def image_path(images_info, height, width):

    for images in images_info:
        h = images["height"]
        w = images["width"]

        images["height"] = height
        images["width"] = width
        for obj in images["object"]:
            obj["xmin"] = int(obj["xmin"] * float(width) / w)
            obj["xw"] = int(obj["xw"] * float(width) / w)

            obj["ymin"] = int(obj["ymin"] * float(height) / h)
            obj["yh"] = int(obj["yh"] * float(height) / h)

        train_image_paths.append(os.path.join(anchor.train_dir, images["filename"]))
        enc_obj.append(images["object"])


image_path(anchor.train_images_anno, config.IMAGE_H, config.IMAGE_W)


train_images = np.array(train_image_paths)
enc_obj = np.array(enc_obj, dtype="object")


def bbox_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    if x3 < x1:
        if x4 < x1:
            return 0

    intersect_x = min(x2, x4) - max(x3, x1)
    intersect_y = min(y2, y4) - max(x3, x1)
    intersection = intersect_x * intersect_y
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    iou = intersection / (box1_area + box2_area - intersection)
    return iou


def best_anchor_box(anchors, box):
    best_iou = -1
    best_anchor = -1
    for k in range(len(anchors)):
        anchor_w, anchor_h = anchors[k]
        anchor = [0, 0, anchor_w, anchor_h]
        iou = bbox_iou(anchor, box)
        if iou > best_iou:
            best_iou = iou
            best_anchor = k

    return best_anchor, best_iou


def bounding_box_encoding(enc_objects, anchors, img_height, img_width, S):  # the encoding is of the form
    # (center_x, center_y, box_width, box_height) each lies in the range (0, S] s=13 in this case
    instance = 0
    i = 0
    for objects in enc_objects:
        for obj in objects:

            center_x = obj["xmin"] + (obj["xw"] / 2)
            center_y = obj["ymin"] + (obj["yh"] / 2)
            center_x = (center_x * S) / float(img_width)
            center_y = (center_y * S) / float(img_height)
            box_width = (obj["xw"] * S) / float(img_width)
            box_height = (obj["yh"] * S) / float(img_height)
            box = [0, 0, box_width, box_height]

            obj_index = np.argmax(tf.cast(np.array(config.LABELS) == obj["name"], dtype="float32"))

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            best_anchor, best_iou = best_anchor_box(anchors, box)

            train_labels[instance, grid_x, grid_y, best_anchor, 1:5] = center_x, center_y, box_width, box_height
            train_labels[instance, grid_x, grid_y, best_anchor, 0] = 1  # true box objectness
            train_labels[instance, grid_x, grid_y, best_anchor, 5 + obj_index] = 1
            i += 1

        instance += 1


bounding_box_encoding(enc_obj, anchors, config.IMAGE_H, config.IMAGE_W, config.GRID_H)

# print(train_images[3])
#
# print(train_labels[3, ..., 1:5][train_labels[3, ..., 0] == 1],
#       "\n", len(train_labels[3, ..., 1:5][train_labels[3, ..., 0] == 1]))
#
# print(train_labels[3, 2, 11, :, 1:5])
#
# print(train_labels[3, ..., 5:][train_labels[3, ..., 0] == 1])
#
# print(len(enc_obj[3]))
#
# print(enc_obj[3])
#
#
# image, _ = image_preprocess(train_images[3], _)
# draw_boxes(image, train_labels[3])

assert not np.any(np.isnan(train_labels))