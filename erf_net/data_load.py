# Author Lingge Li from XJTU(446049454@qq.com)
# data_load function

import sys
import os
import random
import tensorflow as tf

images_path = '/home/llg/Documents/cityscapes/trainImages.txt'
labels_path = '/home/llg/Documents/cityscapes/trainLabels.txt'
'''
def get_filename_list(images_folder, labels_folder):
    images_list = []
    labels_list = []
    if not images_folder:
        raise OSError('Please input images_folder!')
    else:
        for i in os.listdir(images_folder):
            images_path = os.path.join(images_folder, i)
            images_list.append(images_path)
        images.sort()
    if not labels_folder:
        raise OSError('Please input labels_folder!')
    else:
        for i in os.listdir(labels_folder):
            labels_path = os.path.join(labels_folder, i)
            labels_list.append(labels_path)
        labels_list.sort()
    
    # np.vstack(images_list)
    # np.vstack(labels_list)
    
    return images_list, labels_list
'''

def get_filename_list(images_path, labels_path):
    if not images_path:
        raise OSError('Please input images_path!')
    else:
        images_dir = os.path.dirname(images_path)
        images_infile = open(images_path, 'r')
        images_list = []
        for line in images_infile:
            line = line.strip()
            images_list.append('{}/{}'.format(images_dir, line))

    if not labels_path:
        raise OSError('Please input labels_path!')
    else:
        labels_dir = os.path.dirname(labels_path)
        labels_infile = open(labels_path, 'r')
        labels_list = []
        for line in labels_infile:
            line = line.strip()
            labels_list.append('{}/{}'.format(labels_dir, line))

    return images_list, labels_list

#images_list, labels_list = get_filename_list(images_path, labels_path)
#print(images_list)

def instance_generator(sample_path):
    image_fnames, _ = get_filename_list(sample_path, None)

    for fname in image_fnames:
        img = np.array(skimage.io.imread(fname), np.float32)
        #img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_CUBIC) #add 20190627
        yield (img, fname)

class TFLoadingPipeline(object):

    def __init__(self, shuffle = True, num_threads = 1):
        self.shuffle = shuffle
        self.num_threads = num_threads

        self.sess = None
        self.coord = None
        self.threads = None

    def setup(self, images_path, labels_path, batch_size, capacity):
        print('setting up the tf data loading pipeline')
        image_batch_ph, label_batch_ph = self.gen_image_label_batch_ph(batch_size, images_path, labels_path, capacity)
        self._image_batch_ph = image_batch_ph
        self._label_batch_ph = label_batch_ph

    def setup_validate(self, val_images_path, val_labels_path, val_bacth_size, capacity):
        print('setting up the tf loading pipeline for validation data')
        val_images_batch_ph, val_labels_batch_ph = \
            self.gen_image_label_batch_ph(val_bacth_size, val_images_path, val_labels_path, capacity)
        self._val_images_batch_ph = val_images_batch_ph
        self._val_labels_batch_ph = val_labels_batch_ph

    def load_batch(self):
        image_batch, label_batch = self.sess.run([self._image_batch_ph, self._label_batch_ph])
        return image_batch, label_batch
    
    def load_validate_batch(self):
        val_images_batch, val_labels_batch = self.sess.run([self._val_images_batch_ph, self._val_labels_batch_ph])
        return val_images_batch, val_labels_batch

    def gen_image_label_batch_ph(self, batch_size, images_path, labels_path, capacity):
        image_fnames, label_fnames = get_filename_list(images_path, labels_path)
        #print(images_fnames)
        image_fname_tensor = tf.convert_to_tensor(image_fnames, dtype=tf.string)
        label_fname_tensor = tf.convert_to_tensor(label_fnames, dtype=tf.string)

        filename_queue = tf.train.slice_input_producer([image_fname_tensor, label_fname_tensor], shuffle=self.shuffle)

        image_filename = filename_queue[0]
        label_filename = filename_queue[1]

        imageValue = tf.io.read_file(image_filename)
        labelValue = tf.io.read_file(label_filename)

        image_bytes = tf.image.decode_image(imageValue,channels=3)
        
        label_bytes = tf.image.decode_image(labelValue,channels=1)

        image_reshape = tf.reshape(image_bytes, (1024, 2048, 3))
        label_reshape = tf.reshape(label_bytes, (1024, 2048, 1))

        # get tensor
        image = tf.cast(image_reshape, tf.float32)
        #print(type(image), image.get_shape().as_list())
        label = tf.cast(label_reshape, tf.int32)

        print('Filling queue with %d images before starting the pipeline. ' 'This will take a few minutes.' % capacity)
        if self.shuffle:
            image_batch_ph, label_batch_ph = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=self.num_threads,
                capacity=capacity + 3 * batch_size,
                min_after_dequeue=capacity)
        else:
            image_batch_ph, label_batch_ph = tf.train.batch(
                [image, label], 
                batch_size=batch_size,
                num_threads=self.num_threads,
                capacity=capacity + 3 * batch_size)
        print('successful!')
        return image_batch_ph, label_batch_ph

    def start(self, sess):
        if self.sess:
            print('Loading pipeline can only start once!')
            return
        print('start data loading pipeline')
        self.sess = sess
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)

    def shutdown(self):
        print('shut down data loading pipeline')
        self.coord.request_stop()
        self.coord.join(self.threads)



