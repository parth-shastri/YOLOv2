import wget
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_extraction import train_labels, train_images, image_preprocess
from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import time
from utils import calculate_iou, draw_boxes, non_max_suppression, mean_average_precision
import config
import datetime


#  print("DOWNLOADING THE WEIGHTS......")

#  wget.download("https://pjreddie.com/media/files/yolov2.weights", os.getcwd())
#  -- get the weight files

ANCHORS = np.array(config.ANCHORS)


print(train_labels.shape, train_images)

weight_path = r"C:\Users\shast\YOLO_self\yolov2.weights"


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4


def space_to_depth_x2(x):
    return tf.nn.space_to_depth(x, block_size=2)


input_image = Input(shape=(config.IMAGE_H, config.IMAGE_W, 3))


#  Layer 1
x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#  Layer 2
x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
x = BatchNormalization(name='norm_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#  Layer 3
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
x = BatchNormalization(name='norm_3')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 4
x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
x = BatchNormalization(name='norm_4')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 5
x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
x = BatchNormalization(name='norm_5')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#  Layer 6
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 7
x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
x = BatchNormalization(name='norm_7')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 8
x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
x = BatchNormalization(name='norm_8')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#  Layer 9
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
x = BatchNormalization(name='norm_9')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 10
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
x = BatchNormalization(name='norm_10')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 11
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
x = BatchNormalization(name='norm_11')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 12
x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
x = BatchNormalization(name='norm_12')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 13
x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
x = BatchNormalization(name='norm_13')(x)
x = LeakyReLU(alpha=0.1)(x)

skip_connection = x

x = MaxPooling2D(pool_size=(2, 2))(x)
#  Layer 14
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
x = BatchNormalization(name='norm_14')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 15
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
x = BatchNormalization(name='norm_15')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 16
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
x = BatchNormalization(name='norm_16')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 17
x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
x = BatchNormalization(name='norm_17')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 18
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
x = BatchNormalization(name='norm_18')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 19
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
x = BatchNormalization(name='norm_19')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 20
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
x = BatchNormalization(name='norm_20')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 21
skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
skip_connection = BatchNormalization(name='norm_21')(skip_connection)
skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
skip_connection = Lambda(space_to_depth_x2)(skip_connection)

x = concatenate([skip_connection, x])
#  Layer 22
x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
x = BatchNormalization(name='norm_22')(x)
x = LeakyReLU(alpha=0.1)(x)
#  Layer 23
x = Conv2D(config.BOX * (4 + 1 + config.CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
output = Reshape((config.GRID_H, config.GRID_W, config.BOX, 4 + 1 + config.CLASS))(x)
#  small hack to allow true_boxes to be registered when Keras build the model
#  for more information: https://github.com/fchollet/keras/issues/2790
# output = Lambda(lambda args: args[0])([output, true_boxes])

model = Model(input_image, output)

print(model.summary())

def set_pretrained_weights():
    weight_reader = WeightReader(weight_path)

    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv+1):
        conv_layer = model.get_layer('conv_' + str(i))

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta  = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean  = weight_reader.read_bytes(size)
            var   = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    layer = model.layers[-2]

    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape)/(config.GRID_H*config.GRID_W)
    new_bias = np.random.normal(size=weights[1].shape)/(config.GRID_H*config.GRID_W)

    layer.set_weights([new_kernel, new_bias])


print(train_labels[..., 1:3].shape)


def custom_loss(y_true, y_pred):

    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(config.GRID_W), [config.GRID_H]), (1, config.GRID_H, config.GRID_W, 1, 1)), dtype=tf.float32)
    cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

    cell_grid = tf.tile(tf.concat([cell_y, cell_x], -1), [y_pred.shape[0], 1, 1, 4, 1])

    obj = y_true[..., 0] == 1
    noobj = y_true[..., 0] == 0

    """Adjust the predictions"""

    pred_xy = tf.sigmoid(y_pred[..., 1:3]) + cell_grid
    pred_wh = tf.cast(tf.exp(y_pred[..., 3:5]), dtype=tf.float32) *\
              tf.cast(tf.reshape(ANCHORS, shape=[1, 1, 1, config.BOX, 2]), dtype=tf.float32)

    pred_conf = tf.sigmoid(y_pred[..., 0:1])
    pred_class = y_pred[..., 5:][obj]  # consider only those boxes that contain object

    """Adjust Trues"""

    true_xy = y_true[..., 1:3]
    true_wh = y_true[..., 3:5]
    true_conf = y_true[..., 0:1]
    true_class = tf.argmax(y_true[..., 5:], axis=-1)[obj]
    #  Object Loss
    '''iou calculation'''

    iou_scores = calculate_iou(tf.concat([pred_xy, pred_wh], axis=-1), tf.concat([true_xy, true_wh], axis=-1))

    iou_scores = tf.expand_dims(iou_scores, -1)

    i = iou_scores[obj]

    best_ious = tf.reduce_max(iou_scores, -1)

    iou_scores = tf.stop_gradient(iou_scores)

    #  obj_loss = tf.reduce_sum(tf.square((true_conf*iou_scores)[obj] - pred_conf[obj]))
    #  obj_loss = tf.cast(obj_loss, dtype=tf.float32) / (tf.reduce_sum(tf.cast(obj, tf.float32)) + 1e-6)
    obj_loss = tf.losses.BinaryCrossentropy()((true_conf*iou_scores)[obj], pred_conf[obj])

    #  No-Object Loss
    iou_condition = tf.expand_dims(tf.cast(best_ious < 0.6, dtype=tf.float32), -1)
    # noobj = tf.math.logical_and(noobj, best_ious < 0.6)
    #  noobj_loss = tf.square(y_true[..., 0:1][noobj] - y_pred[..., 0:1][noobj])
    #  noobj_loss = tf.reduce_sum(noobj_loss) / (tf.reduce_sum(tf.cast(noobj, tf.float32)) + 1e-6)
    noobj_loss = tf.losses.BinaryCrossentropy()(true_conf[noobj], pred_conf[noobj])

    #  Coordinate loss
    #  xy_loss = tf.reduce_sum(tf.square(true_xy[obj] - pred_xy[obj]))
    xy_loss = tf.losses.MeanSquaredError()(true_xy[obj], pred_xy[obj])
    # wh_loss = tf.reduce_sum(tf.square(tf.math.sqrt(true_wh[obj]) - tf.math.sqrt(pred_wh[obj])))
    wh_loss = tf.losses.MeanSquaredError()(tf.math.sqrt(true_wh[obj]), tf.math.sqrt(pred_wh[obj]))
    # coord_loss = tf.cast(xy_loss + wh_loss, dtype=tf.float32) / (tf.reduce_sum(tf.cast(obj, tf.float32)) + 1e-6)
    coord_loss = (xy_loss + wh_loss) / 2.0

    #  Class loss

    class_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)(true_class, pred_class)
    # class_loss = tf.reduce_sum(class_loss) / (tf.reduce_sum(tf.cast(obj, tf.float32)) + 1e-6)

    nb_true_boxes = tf.reduce_sum(y_true[..., 0])
    nb_pred_boxes = tf.reduce_sum(tf.cast(tf.squeeze((true_conf * iou_scores), -1) > 0.5, dtype=tf.float32) *
                                  tf.cast(tf.squeeze(pred_conf, -1) > 0.3, dtype=tf.float32))

    recall = nb_pred_boxes / (1e-6 + nb_true_boxes)   # Batch Recall
    total_recall = 0.0
    total_recall += recall

    loss = config.COORD_SCALE * coord_loss + config.CLASS_SCALE * class_loss +\
            config.OBJECT_SCALE * obj_loss + config.NO_OBJECT_SCALE * noobj_loss

    return loss, total_recall, class_loss, coord_loss, obj_loss, noobj_loss


set_pretrained_weights()


train_data = tf.data.Dataset.from_tensor_slices((train_images[:2300], train_labels[:2300]))
train_data = train_data.map(lambda x, y: tf.py_function(image_preprocess, [x, y], (tf.float32, tf.float32)),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(config.BATCH_SIZE)
val_data = tf.data.Dataset.from_tensor_slices((train_images[2300:2500], train_labels[2300:2500]))
val_data = val_data.map(lambda x, y: tf.py_function(image_preprocess, [x, y], (tf.float32, tf.float32)),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(config.BATCH_SIZE)

print(train_data.element_spec)

for image, label in train_data.take(1):
    out = model.predict(image)
    print(out.shape)

    loss = custom_loss(tf.cast(train_labels[:config.BATCH_SIZE, ...], dtype=tf.float32), tf.convert_to_tensor(out))
    print(loss)


early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.001,
                           patience=10,
                           mode='min',
                           verbose=1)

if config.OPTIMIZER.upper() == "ADAM":
    optimizer = Adam(lr=config.LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000)

if config.OPTIMIZER.upper() == "SGD":
    optimizer = SGD(lr=1e-2, decay=0.0005, momentum=0.9)

if config.OPTIMIZER.upper() == "RMSPROP":
    optimizer = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08, decay=0.0)

ckpt_dir = "ckpt-overfit"
ckpt_prefix = os.path.join(ckpt_dir, "yolo_ckpt")
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print("Restored!")

#  model.compile(loss=custom_loss, optimizer=optimizer)
#  model.load_weights("weights_coco_2500.h5")
#  model.fit(train_data,
#            epochs=100, verbose=1, callbacks=[early_stop, checkpoint, tensorboard], validation_data=val_data)


'''Custom training (without using .fit())'''

#  log_dir = "logs/"
# 
#  summary_writer = tf.summary.create_file_writer(
#    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))    #  for writing the Summary in Tensorboard
# 
# 
#  @tf.function
#  def train_step(image_batch, label_batch):
# 
#      with tf.GradientTape() as tape:
#          y_pred = model(image_batch, training=True)
#          loss, recall, class_loss, coord_loss, obj_loss, noobj_loss = custom_loss(label_batch, y_pred)
# 
#      model_gradients = tape.gradient(loss, model.trainable_variables)
#      optimizer.apply_gradients(zip(model_gradients, model.trainable_variables))
#      with summary_writer.as_default():
#          tf.summary.scalar('total_loss', loss, step=epoch)
#          tf.summary.scalar('gen_gan_loss', coord_loss, step=epoch)
#          tf.summary.scalar('recall', recall, step=epoch)
#          tf.summary.scalar('class_loss', class_loss , step=epoch)
# 
#      return loss, recall, class_loss, coord_loss, obj_loss, noobj_loss
#  epochs = 30
# 
#  for epoch in range(epochs):
#      start = time.time()
#      n = 0
# 
#      for images, labels in train_data:
#          loss, recall, class_loss, coord_loss, obj_loss, noobj_loss = train_step(images, labels)
#          if n % 10 == 0:
#              print(".", end='')
#          n += 1
#          if n % 5 == 0:
#              print("Training loss : {} Recall : {} Obj_loss + Noobj_loss : {} Coord_loss : {} Class_loss : {}"
#                    .format(loss, recall, (obj_loss + noobj_loss), coord_loss, class_loss))
# 
#      if (epoch+1) % 5==0:
#          ckpt_save_path = manager.save()
#          print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
#                                                              ckpt_save_path))
#      print("Time taken for epoch {} is {} sec\n".format(epoch+1, time.time()-start))


'''Inference of the model'''


def yolo_inf():
    for image, label in train_data.take(1):

        y_pred = model.predict(image)
        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(config.GRID_W),
                                            [config.GRID_H]), (1, config.GRID_H, config.GRID_W, 1, 1)), dtype=tf.float32)
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        cell_grid = tf.tile(tf.concat([cell_y, cell_x], -1), [config.BATCH_SIZE, 1, 1, 4, 1])

        obj = label[..., 0] == 1

        """Adjust the predictions"""

        pred_xy = tf.sigmoid(y_pred[..., 1:3]) + cell_grid
        pred_wh = tf.cast(tf.exp(y_pred[..., 3:5]), dtype=tf.float32) * tf.cast(
            tf.reshape(ANCHORS, shape=[1, 1, 1, config.BOX, 2]), dtype=tf.float32)

        pred_conf = tf.sigmoid(y_pred[..., 0:1])
        pred_conf_ = pred_conf[obj]
        pred_class = tf.nn.softmax(tf.cast(y_pred[..., 5:], tf.float32), axis=-1)
        pred_class_ = tf.argmax(pred_class[tf.sigmoid(y_pred[..., 0]) > 0.45], axis=-1)

        pred = tf.concat([pred_conf, pred_xy, pred_wh, pred_class], axis=-1)

        pred_coord = tf.concat([pred_xy, pred_wh], axis=-1)[obj]

        label_coord_ = label[..., 1:5][obj]
        label_class_ = tf.expand_dims(tf.argmax(label[..., 5:][obj], axis=-1), axis=-1)
        label_boxes_ = tf.concat([label[..., 0:1][obj], label_coord_, tf.cast(label_class_, tf.float32)], axis=-1)
        mAP = 0

        for i in range(len(image)):
            nms_boxes = non_max_suppression(pred[i])
            label_coord = label[i, ..., 1:5][label[i, ..., 0]==1]
            label_class = tf.expand_dims(tf.argmax(label[i, ..., 5:][label[i, ..., 0] == 1], axis=-1), axis=-1)
            label_boxes = tf.concat([label[i, ..., 0:1][label[i, ..., 0]==1], label_coord, tf.cast(label_class, tf.float32)], axis=-1)
            mAP += mean_average_precision(nms_boxes, label_boxes)
            draw_boxes(image[i], nms_boxes)
        print("mAP on the given set of examples : {}\n".format(mAP / len(image)))


yolo_inf()