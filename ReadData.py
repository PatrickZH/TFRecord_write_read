import numpy as np
import tensorflow as tf
import sys
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist
import matplotlib.pyplot as plt
sys.path.append("/home/zhaobo/TFData/models/slim")
# import nets.vgg as vgg
# vgg = models.slim.nets.vgg
from nets import vgg


def read_tfrecord(filepath):
    filename_queue = tf.train.string_input_producer(filepath)
    reader = tf.TFRecordReader()
    _, examples = reader.read(filename_queue)
    features = tf.parse_single_example(
        examples, features={'image': tf.FixedLenFeature([], tf.string),
                            'label': tf.FixedLenFeature([], tf.int64)
                            }
    )

    num = -1
    image_shape = [224, 224, 3]

    labels = tf.cast(features['label'], tf.int32)
    # labels = tf.cast(labels, tf.int32)

    images = tf.decode_raw(features['image'], tf.uint8)
    images = tf.reshape(images, image_shape)
    images = tf.cast(images, tf.float32)

        # plt.imshow(images)
    # labels = tf.decode_raw(features['label'], tf.int64)

    # init_op = tf.initialize_all_variables()
    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     images_eval = images.eval()
    # return images, labels
    return images, labels



# def vgg16(inputs):
#   with slim.arg_scope([slim.conv2d, slim.fully_connected],
#                       activation_fn=tf.nn.relu,
#                       weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
#                       weights_regularizer=slim.l2_regularizer(0.0005)):
#     net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
#     net = slim.max_pool2d(net, [2, 2], scope='pool1')
#     net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
#     net = slim.max_pool2d(net, [2, 2], scope='pool2')
#     net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
#     net = slim.max_pool2d(net, [2, 2], scope='pool3')
#     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
#     net = slim.max_pool2d(net, [2, 2], scope='pool4')
#     net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
#     net = slim.max_pool2d(net, [2, 2], scope='pool5')
#     net = slim.fully_connected(net, 4096, scope='fc6')
#     net = slim.dropout(net, 0.5, scope='dropout6')
#     net = slim.fully_connected(net, 4096, scope='fc7')
#     net = slim.dropout(net, 0.5, scope='dropout7')
#     net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
#   return net

# print(type(images))





filepath = ['/home/zhaobo/FeatExtr/test/CUB']
images, labels = read_tfrecord(filepath)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(1):
        img, l= sess.run([images, labels])
        img_real = np.reshape(img, (224,224,3))

        print(img_real)
        plt.imshow(img_real)
        plt.show()


# images = tf.Variable(tf.random_normal([100,
#                                        224,
#                                        224, 3],
#                                       dtype=tf.float32,
#                                       stddev=1e-1))
# labels = tf.ones([100], tf.int32)


#
# num = 200
# train_log_dir = '/home/zhaobo/FeatExtr/test/'
# init = tf.initialize_all_variables()
#
#
#
#
# filepath = ['/home/zhaobo/FeatExtr/test/CUB']
#
#
# with tf.Graph().as_default():
#     images, labels = read_tfrecord(filepath)
#     logits = mnist.inference(images)



# sess = tf.Session()
# sess.run(init)
# threads = tf.train.start_queue_runners(sess=sess)
# coord = tf.train.Coordinator()
# # labels = tf.reshape(labels, [num])
# # print(labels.get_shape())
#
#
# # images = tf.cast(images, tf.float32)
# # images = tf.reshape(images, [-1, 224, 224, 3])
#
# # labels = tf.cast(labels, tf.int32)
# # labels = tf.reshape(labels, [-1])
# #
# #
# # predictions, end_points = vgg.vgg_19(images, num_classes=num, is_training=True)
# predictions = vgg16(images)
#
# onehot_labels = tf.one_hot(labels, num)
# #
# # print(onehot_labels.get_shape())
# #
# #
# loss = slim.losses.softmax_cross_entropy(predictions, onehot_labels)
# #
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
# train_tensor = slim.learning.create_train_op(loss, optimizer)
# slim.learning.train(train_tensor, train_log_dir)


























