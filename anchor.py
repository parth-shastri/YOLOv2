import tensorflow as tf
import os
import numpy as np
import json
import tqdm
import matplotlib.pyplot as plt
import config

anno_dir = config.ANNO_DIR
train_dir = config.TRAIN_DIR
anno_file = os.path.join(anno_dir, "instances_train2014.json")

train_images_info = []
LABELS = {}


def parse_annotations(anno_file):

    #n = int(input("The number of training examples:"))
    n = 2500
    #new = input("Do you want to sample the data again(y/n):")
    new = 'n'
    with open(anno_file, "r") as file:
        instance = json.load(file)

    for category in instance["categories"]:

        LABELS[category["name"]] = category["id"] - 1

    print(LABELS)
    if new == "y":
        for image in tqdm.tqdm(instance["images"][: n]):

            img = {"object": []}

            img["filename"] = image["file_name"]

            if not os.path.exists(os.path.join(train_dir,image["file_name"])):
                    print("File doesnt exist\n")

            img["height"] = image["height"]
            img["width"] = image["width"]

            for annot in instance["annotations"]:

                obj = dict()

                if image["id"] == annot["image_id"]:

                    obj["name"] = LABELS[annot["category_id"]]
                    obj["xmin"] = annot["bbox"][0]
                    obj["ymin"] = annot["bbox"][1]
                    obj["xw"] = annot["bbox"][2]
                    obj["yh"] = annot["bbox"][3]

                    img["object"].append(obj)

            train_images_info.append(img)
        print(train_images_info[2344])
        # with open("labeles_created.json", "w") as f:
        #
        #     json.dump(train_images_info, f)

    else:
        pass

parse_annotations(anno_file)

sampled_ = "labeles_created.json"
if not os.path.exists(sampled_):
    raise NotImplemented

with open("labeles_created.json", "r") as f:

    train_images_anno = json.load(f)

wh = []
for image in train_images_anno:

    wa = float(image["width"])
    ha = float(image["height"])
    for obj in image["object"]:
        w = obj["xw"] / wa
        h = obj["yh"] / ha
        temp = [w, h]
        wh.append(temp)

wh = np.array(wh)

# plt.figure(figsize=(10, 10))
# plt.scatter(wh[:, 0], wh[:, 1])
# plt.title("clusters")
# plt.xlabel("normalized width")
# plt.ylabel("normalized height")
#plt.show()


def iou(box, clusters):
    # shape of clusters is (rows, 2)

    x = np.minimum(clusters[:, 0], box[0])  # returns the tensor with shape (rows,) containing the width
    y = np.minimum(clusters[:, 1], box[1])  # returns the tensor with shape (rows,) containing the height

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0]*clusters[:, 1]
    iou_ = intersection/(box_area + cluster_area - intersection)

    return iou_  # shape = shape of the clusters


def kmeans(boxes, k, dist=np.median, seed=1):

    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed(seed)

    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for icluster in range(k):
            distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

        nearest_clusters = np.argmin(distances, axis=1)  # returns the vector of shape (rows,)
        # containing the indices of the clusters assigned to each point

        if (last_clusters == nearest_clusters).all():  # if and only if all the elements match
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters, nearest_clusters, distances


kmax = 11
dist = np.mean
results = {}
# for k in range(2,kmax):
#     clusters, nearest_clusters, distances = kmeans(wh, k, seed=2, dist=dist)
#     WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]),nearest_clusters])
#     result = {"clusters":             clusters,
#               "nearest_clusters":     nearest_clusters,
#               "distances":            distances,
#               "WithinClusterMeanDist": WithinClusterMeanDist}
#     print("{:2.0f} clusters: mean IoU = {:5.4f}".format(k,1-result["WithinClusterMeanDist"]))
#     results[k] = result
#
# final anchor boxes selscted == 4
#
# clusters, nearest_clusters, distances = kmeans(wh, k=4, seed=2, dist=dist)
#





