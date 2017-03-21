import numpy as np
import tensorflow as tf
from PIL import Image
import os


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(bytes_list=tf.train.FloatList(value=[value]))

def Read_images_SaveTFRecord(inputpath, shape, outputpath):
    # reader = tf.WholeFileReader()
    writer = tf.python_io.TFRecordWriter(outputpath)
    filelist = os.listdir(inputpath)
    label_dict = dict()
    num = -1
    for folder in filelist:
        str_folder = inputpath+'/'+folder
        if os.path.isdir(str_folder):
            num += 1
            print('Processing No. '+str(num)+' ... '+str_folder)
            imagelist = os.listdir(str_folder)
            label_dict[num] = folder
            for each in imagelist:
                str_image = str_folder+'/'+each
                # print(str_image)
                try:
                    image = Image.open(str_image)
                except OSError:
                    continue

                image = image.resize([shape,shape])
                # example = tf.train.Example(features=tf.train.Features(feature={
                #     "label": _int64_feature(int(num)),
                #     "image": _bytes_feature(image.tobytes())
                # }))
                image = image.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[num])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))
                writer.write(example.SerializeToString())
    writer.close()

# inputpath = ['/media/zhaobo/Seagate4T2/ImageNet/ILSVRC2012/ILSVRC2012_img_train/n02058221/n02058221_12.JPEG',
#              '/media/zhaobo/Seagate4T2/ImageNet/ILSVRC2012/ILSVRC2012_img_train/n02058221/n02058221_756.JPEG']
outputpath = '/home/zhaobo/FeatExtr/test/CUB'
shapes_vgg = 224
inputpath = '/media/zhaobo/Seagate4T2/CUB/CUB_200_2011/CUB_200_2011/images'
Read_images_SaveTFRecord(inputpath, shapes_vgg, outputpath)






                # filename_queue = tf.train.string_input_producer(str)
        # key, value = reader.read(filename_queue)
        # my_img = tf.image.decode_jpeg(value)
        # init_op = tf.initialize_all_variables()
        # with tf.Session() as sess:
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(coord=coord)
            # sess.run(init_op)
            # image = my_img.eval()
            # image = image.eval()
            # print(image.shape)
            #
            # image = tf.image.resize_images(image, [shapes_vgg, shapes_vgg])
            # print(image.shape)
            # example = tf.train.Example(features=tf.train.Features(feature={
            #     'height': _int64_feature(shapes_vgg),
            #     'width': _int64_feature(shapes_vgg),
            #     'depth': _int64_feature(3),
            #     'label': _int64_feature(int(label)),
            #     'image_raw': _float_feature(image)}))
            # writer.write(example.SerializeToString)

    # coord.request_stop()
    # coord.join(threads)










