# Author Lingge Li from XJTU(446049454@qq.com)
# inference

import tensorflow as tf
import os
import time
import cv2
import numpy as np
import sys

from erf_net import ERF_Net
from erf_net_slim import *
from train_slim import *
from data_load import instance_generator

from config import cfg
from IOU_metrics import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model', 'erf', """ which model to use """)
tf.app.flags.DEFINE_string('ckpt_path', '', """ path to restore model ckpt """)
tf.app.flags.DEFINE_string('images_path', '', """ path to train image """)
tf.app.flags.DEFINE_string('output_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_string('mask_path', '', """ path to validate label image """)
tf.app.flags.DEFINE_boolean('restore_avg', True, """ whether to restore moving avg version """)

def checkArgs():
    print('Inference model: {}'.format(FLAGS.model))
    print("Train image path: {}".format(FLAGS.images_path))
    print("Prediction output path: {}".format(FLAGS.output_path))
    print("Mask output path: {}".format(FLAGS.mask_path))
    print("Modle checkpoint path: {}".format(FLAGS.ckpt_path))

def genPredProb(image, num_classes):
    """ store label data to colored image """
    im = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(num_classes - 1):
        im[:,:,0] += image[:,:,i+1] * cfg.COLOR_CODE[i][0]
        im[:,:,1] += image[:,:,i+1] * cfg.COLOR_CODE[i][1]
        im[:,:,2] += image[:,:,i+1] * cfg.COLOR_CODE[i][2]
        #r = np.zeros_like(prob_img)
        #g = prob_img.copy() * 255
        #b = np.zeros_like(prob_img)
        #rgb = np.zeros((image.shape[0], image.shape[1], 3))
        #rgb[:,:,0] = r
        #rgb[:,:,1] = g
        #rgb[:,:,2] = b
    im = np.uint8(im)
    return im

def main(self):
    checkArgs()

    output_path =  FLAGS.output_path
    mask_path = FLAGS.mask_path

    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    if not os.path.isdir(mask_path):
        os.makedirs(mask_path)
    
    if FLAGS.model == 'erf':
        model = ERF_Net()
    else:
        print('please input a model!')
        sys.exit(-1)
    
    if FLAGS.restore_avg:
        # get moving avg
        variable_averages = tf.train.ExponentialMovingAverage(0.99)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
    else:
        saver = tf.train.Saver(variables_to_restore)
    

    with tf.Session() as sess:
        # restore model
        print('restoring model ......')
        saver.restore(sess, FLAGS.ckpt_path)
        total_time_elapsed = 0.0

        for image, fname in instance_generator(FLAGS.images_path):
            print('predicting for {}'.format(fname))

            feed_dict = {
                model.input_image: [image],
                model.is_training: True
            }
            '''
	        y = sess.run(tf.get_collection('l1_down'), feed_dict = feed_dict) # 2019.10.15
                print(len(y[0]), len(y[0][0]), len(y[0][0][0]), len(y[0][0][0][0]))
            
            y = np.array(y[0][0])
            #print('downsampler output : ', type(y), y.shape, len(y.tolist()))
	        print(y.shape[2])
            length = y.shape[2]

            for num in range(length):
                x = y[:, :, num] * 255
                cv2.imwrite('/home/llg/Documents/erf_net/save_image/{}.png'.format(num), x)
            print('over!')
            no
            '''
            '''
            y = sess.run(tf.get_collection('deconv22'), feed_dict = feed_dict) # 2019.10.15
            print(len(y[0]), len(y[0][0]), len(y[0][0][0]), len(y[0][0][0][0]))
            
            y = np.array(y[0][0])
            #print('downsampler output : ', type(y), y.shape, len(y.tolist()))
	        #print(y.shape[2])
            length = y.shape[2]
            for num in range(length):
                x = y[:, :, num] * 255
                cv2.imwrite('/home/llg/Documents/erf_net/save_image/{}.png'.format(num + 100), x)
            print('over!')
            '''
            begin_time = time.time()
            prediction = sess.run(model.segment_output, feed_dict=feed_dict)
            end_time = time.time()

            print('cost time: {} s'.format(end_time - begin_time))
            total_time_elapsed = total_time_elapsed + end_time - begin_time

            # output_image
            output_fname = output_path + '/' + os.path.basename(fname)
            pred_prob = genPredProb(prediction[0], cfg.NUM_CLASSES)
            flag = cv2.imwrite(output_fname, pred_prob)
            if not flag:
                print('writing image to {} failed!'.format(fname))
                sys.exit(-1)

            # mask image
            mask_fname = mask_path + '/' + os.path.basename(fname)
            r, g, b = cv2.split(image.astype(np.uint8))
            cv_img = cv2.merge([b, g, r])
            masked = cv2.addWeighted(cv_img, 1.0, pred_prob, 1.0, 0)
            flag = cv2.imwrite(mask_fname, masked)
            if not flag:
                print('writing image to {} failed!'.format(mask_fname))
                sys.exit(-1)
        
        print('total time elapsed: {} s'.format(total_time_elapsed))

if __name__ == '__main__':
    tf.app.run()
