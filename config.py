LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']

ANCHORS = [[0.61917456, 0.9282806],
           [9.98163656, 9.16328879],
           [2.17244281, 2.80338593],
           [4.32698054, 6.53312182]]


ANNO_DIR = r"C:\Users\shast\datasets\COCOtrain2014\annotations"
TRAIN_DIR = r"C:\Users\shast\datasets\COCOtrain2014"

IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 4
CLASS            = 80
OBJ_THRESHOLD    = 0.3  #0.5
NMS_THRESHOLD    = 0.3  #0.45

NO_OBJECT_SCALE  = 5.0
OBJECT_SCALE     = 1.0
COORD_SCALE      = 5.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 8
WARM_UP_BATCHES  = 0

LEARNING_RATE = 3e-4
OPTIMIZER = "adam"

