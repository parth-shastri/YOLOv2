import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import config


def calculate_iou(T1, T2):   # Take tensor inputs of shape (B_S, S, S, B, 4)

    t1_xy = T1[..., 0:2]
    t1_wh = T1[..., 2:4]

    t1_wh_half = t1_wh / 2.
    t1_mins = t1_xy - t1_wh_half
    t1_maxes = t1_wh_half + t1_xy

    t2_xy = T2[..., 0:2]
    t2_wh = T2[..., 2:4]

    t2_wh_half = t2_wh / 2.
    t2_mins = t2_xy - t2_wh_half
    t2_maxes = t2_wh_half + t2_xy

    intersect_mins = tf.maximum(t1_mins, t2_mins)
    intersect_maxes = tf.minimum(t1_maxes, t2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)

    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = t1_wh[..., 0] * t1_wh[..., 1]
    pred_areas = t2_wh[..., 0] * t2_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = tf.math.truediv(intersect_areas, union_areas)

    return iou_scores


""" An utility function to draw bounding boxes on an image"""


def draw_boxes(image, predictions):
    plt.figure(figsize=(15, 15))

    import seaborn as sns
    color_palette = list(sns.xkcd_rgb.values())
    iobj = 0

    if type(predictions) == list:

        for box in predictions:
            x, y, w, h = box[1:5]
            class_arg = int(box[5])
            class_nm = config.LABELS[class_arg]
            mult_x = config.IMAGE_W / config.GRID_W
            mult_y = config.IMAGE_H / config.GRID_H
            c = color_palette[iobj]
            iobj += 1
            xmin = (x - 0.5 * w)
            ymin = (y - 0.5 * h)
            xmax = (x + 0.5 * w)
            ymax = (y + 0.5 * h)

            org1, org2 = int(xmin * mult_x + 3), int(ymin * mult_y + 13)

            # cv2.rectangle(image, (int(xmin*mult_x), int(ymin*mult_y)), (int(xmax * mult_x), int(ymax * mult_y)),
            # (0, 255, 0), 2)
            #
            # cv2.putText(image,
            #             class_nm[0],
            #             (org1, org2),
            #             cv2.FONT_HERSHEY_PLAIN,
            #             0.7,
            #             (0, 255, 0), 1)

            plt.text((xmin * mult_x + 3), (ymin * mult_y + 13),
                     class_nm, color=c, fontsize=12)
            plt.plot(np.array([xmin, xmin]) * mult_x,
                     np.array([ymin, ymax]) * mult_y, color=c, linewidth=2)
            plt.plot(np.array([xmin, xmax]) * mult_x,
                     np.array([ymin, ymin]) * mult_y, color=c, linewidth=2)
            plt.plot(np.array([xmax, xmax]) * mult_x,
                     np.array([ymax, ymin]) * mult_y, color=c, linewidth=2)
            plt.plot(np.array([xmin, xmax]) * mult_x,
                     np.array([ymax, ymax]) * mult_y, color=c, linewidth=2)

    else:
        for i_grid_h in range(config.GRID_H):
            for i_grid_w in range(config.GRID_W):
                for ianchor in range(config.BOX):
                    vec = predictions[i_grid_h, i_grid_w, ianchor, :]
                    C = vec[0]
                    if C > 0.6:
                        class_nm = np.array(config.LABELS)[np.argmax(vec[5:])]
                        x, y, w, h = vec[1:5]
                        mult_x = config.IMAGE_W / config.GRID_W
                        mult_y = config.IMAGE_H / config.GRID_H
                        c = color_palette[iobj]
                        iobj += 1
                        xmin = (x - 0.5 * w)
                        ymin = (y - 0.5 * h)
                        xmax = (x + 0.5 * w)
                        ymax = (y + 0.5 * h)

                        org1, org2 = int(xmin * mult_x + 3), int(ymin * mult_y + 13)

                        # cv2.rectangle(image, (int(xmin*mult_x), int(ymin*mult_y)), (int(xmax * mult_x), int(ymax * mult_y)),
                        # (0, 255, 0), 2)
                        #
                        # cv2.putText(image,
                        #             class_nm[0],
                        #             (org1, org2),
                        #             cv2.FONT_HERSHEY_PLAIN,
                        #             0.7,
                        #             (0, 255, 0), 1)

                        plt.text((xmin * mult_x + 3), (ymin * mult_y + 13),
                                 class_nm, color=c, fontsize=12)
                        plt.plot(np.array([xmin, xmin]) * mult_x,
                                 np.array([ymin, ymax]) * mult_y, color=c, linewidth=2)
                        plt.plot(np.array([xmin, xmax]) * mult_x,
                                 np.array([ymin, ymin]) * mult_y, color=c, linewidth=2)
                        plt.plot(np.array([xmax, xmax]) * mult_x,
                                 np.array([ymax, ymin]) * mult_y, color=c, linewidth=2)
                        plt.plot(np.array([xmin, xmax]) * mult_x,
                                 np.array([ymax, ymax]) * mult_y, color=c, linewidth=2)

    plt.imshow(image)
    plt.show()


def non_max_suppression(y_pred , threshold=0.45, iou_threshold=0.5):  # shape 0f y_pred : (S, S, config.BOX, 5 + C)

    sel_boxes_mask = y_pred[..., 0] > threshold

    selected_boxes = y_pred[..., 0:5][sel_boxes_mask]

    selected_class = tf.expand_dims(tf.argmax(y_pred[..., 5:], axis=-1)[sel_boxes_mask], axis=-1)

    selected_boxes = tf.concat([selected_boxes, tf.cast(selected_class, tf.float32)], axis=-1)

    selected_boxes = list(np.array(selected_boxes))

    sorted_sel_boxes = sorted(selected_boxes, key= lambda x : x[5], reverse=True)

    nms_boxes = []

    while sorted_sel_boxes:

        chosen = sorted_sel_boxes.pop(0)

        sorted_sel_boxes = [box for box in sorted_sel_boxes
                            if box[5] != chosen[5]
                            or
                            calculate_iou(tf.convert_to_tensor(chosen[1:5]), tf.convert_to_tensor(box[1:5])) < iou_threshold]

        nms_boxes.append(chosen)

    return nms_boxes


def mean_average_precision(pred_boxes, true_boxes, iou_threshold = 0.5):   # pred_boxes shape : (no_of boxes, 6)

    """pred_boxes : list of lists [conf, coords, class]
    true_boxes : list of lists [conf, coords, class]"""

    true_boxes = np.array(true_boxes)
    pred_boxes = np.array(pred_boxes)

    epsilon = 1e-6
    average_precisions = []
    for C in range(80):
         detected = []
         ground_truth = []

         for box in true_boxes:
             if box[5] == C:
                 ground_truth.append(box)


         for box in pred_boxes:
             if box[5] == C:
                 detected.append(box)

         total_boxes = len(ground_truth)
         if total_boxes == 0:
             continue

         detected = sorted(detected, key= lambda x: x[0], reverse=True)
         TP = np.zeros((len(detected)))
         FP = np.zeros((len(detected)))
         best_iou = 0

         for det_idx, detection in enumerate(detected):

             for idx, gt_box in enumerate(ground_truth):

                 iou = calculate_iou(tf.convert_to_tensor(detection[1:5]), tf.convert_to_tensor(gt_box[1:5]))

                 if iou > best_iou:
                     best_iou = iou
                     best_gt_index = idx

             if best_iou > iou_threshold:
                 TP[det_idx] = 1
             else:
                 FP[det_idx] = 1


         TP_cumsum = np.cumsum(TP)
         FP_cumsum = np.cumsum(FP)
         recalls = TP_cumsum / (total_boxes + epsilon)
         precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

         precisions = np.concatenate((np.array([1]), precisions))
         recalls = np.concatenate((np.array([0]), recalls))

         average_precisions.append(np.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)



























