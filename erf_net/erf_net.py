# Author Lingge Li from XJTU(446049454@qq.com)
# erf_net

import tensorflow as tf
import numpy as np
from erf_net_slim import *
from config import cfg

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048
IMAGE_DEPTH = 3
BATCH_SIZE = 1
NUM_CLASSES = 30
LABEL_WEIGHTED = True
CLASS_WEIGHT = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
SUBTRACT_CHANNEL_MEAN = False
WEIGHT_DECAY = 2e-4
USE_BN = True
USE_DROPOUT = True
DROP_RATE = 0.5

class ERF_Net(object):
    def __init__(self):

        self.loss_collection = 'losses'

        self.add_input_layer()
        self.add_forward_layer()
        self.add_output_layer()

        self.add_loss_layer()

    def add_input_layer(self):
        image_h = cfg.IMAGE_HEIGHT
        image_w = cfg.IMAGE_WIDTH
        image_c = cfg.IMAGE_DEPTH
        batch_size = cfg.BATCH_SIZE
        with tf.name_scope('input'):
            self.input_image = tf.placeholder(tf.float32, shape = [batch_size, image_h, image_w, image_c], name = 'image')
            self.is_training = tf.placeholder_with_default(False, (), name = 'is_training')
            self.is_train = False
            self.label_image = tf.placeholder(tf.int32, shape = [None, image_h, image_w, 1], name = 'label')

    def add_output_layer(self):
        with tf.name_scope('output'):
            self.segment_output = tf.nn.softmax(self.segment_logit, name = 'prob')
            print('self.segment_output = {}'.format(self.segment_output.get_shape().as_list()))
    
    def add_loss_layer(self):
        num_class = cfg.NUM_CLASSES
        with tf.name_scope('model_loss'):
            regularize_loss = tf.add_n(tf.get_collection(self.loss_collection), name = 'regularize_loss')
            logits = tf.reshape(self.segment_logit, (-1, num_class))
            weighted = cfg.LABEL_WEIGHTED

            # consturct one-hot label array
            #if batch_size = 1, 1024x2048, (2097152,)  (2097152, 19)

            label_flat = tf.reshape(self.label_image, [-1])
            labels = tf.one_hot(label_flat, depth = num_class)

            if weighted:
                class_weight = np.array(cfg.CLASS_WEIGHT, dtype = np.float32)
                epsilon = tf.constant(value = 1e-10)
                logits = logits + epsilon

                softmax = tf.nn.softmax(logits)
                cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), class_weight), axis=[1])
                cross_entropy_loss = tf.reduce_mean(cross_entropy, name = 'segment_loss')
            else:
                # (39845888,) 1024x2048x19
                labels = tf.reshape(labels, [-1])
                # here the labels must be 1 wei
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy_loss = tf.reduce_mean(cross_entropy, name = 'segment_loss')

            tf.add_to_collection(self.loss_collection, cross_entropy_loss)

            self.total_loss = tf.add_n(tf.get_collection(self.loss_collection), name = 'total_loss')
            self.losses = [self.total_loss, cross_entropy_loss, regularize_loss]

    def build_encoder(self, input_data):
        input_shape = input_data.get_shape().as_list()
        print('encoder input shape: [' + ','.join([str(x) for x in input_shape]) + ']')
        
        # downsampler 1
        output = downsampler(input_data, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                             use_bn=cfg.USE_BN, is_training=self.is_training, bn_decay=0.99,
                             loss_collection=self.loss_collection, name='l1_downsampler')
        
        # downsampler 2
        output = downsampler(output, 64, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                             use_bn=cfg.USE_BN, is_training=self.is_training, bn_decay=0.99,
                             loss_collection=self.loss_collection, name='l2_dowansampler')
        
        # nonbt1d_3
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l3_nonbt1d')
        
        # nonbt1d_4
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l4_nonbt1d')
        
        # nonbt1d_5
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l5_nonbt1d')
        
        # nonbt1d_6
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l6_nonbt1d')
        
        # nonbt1d_7
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l7_nonbt1d')
        
        # downsampler 8
        output = downsampler(output, 128, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY,
                             use_bn=cfg.USE_BN, is_training=self.is_training, bn_decay=0.99,
                             loss_collection=self.loss_collection, name='l8_dowansampler')

        # nonbt1d_9(dilated 2)
        output = res_nonbt1d(output, 3, True, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l9_nonbt1d')
        
        # nonbt1d_10(dilated 4)
        output = res_nonbt1d(output, 3, True, 4, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l10_nonbt1d')
        
        # nonbt1d_11(dilated 8)
        output = res_nonbt1d(output, 3, True, 8, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l11_nonbt1d')

        # nonbt1d_12(dilated 16)
        output = res_nonbt1d(output, 3, True, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l2_nonbt1d')
        
        # nonbt1d_13(dilated 2)
        output = res_nonbt1d(output, 3, True, 2, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l13_nonbt1d')
        
        # nonbt1d_14(dialted 4)
        output = res_nonbt1d(output, 3, True, 4, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l14_nonbt1d')
        
        # nonbt1d_15(dilated 8)
        output = res_nonbt1d(output, 3, True, 8, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l15_nonbt1d')
        
        # nonbt1d_16(dilated 16)
        output = res_nonbt1d(output, 3, True, 16, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l16_nonbt1d')
        
        return output
    
    def build_segmenter(self, input_data):
        input_shape = input_data.get_shape().as_list()
        print('segmenter input shape: [' + ','.join([str(x) for x in input_shape]) + ']')

        # upsampled by deconvolution

        # deconvolution 17(upsampling)
        feature_map_h, feature_map_w = get_deconv_output_feature_map_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3)
        deconv_output_shape = [cfg.BATCH_SIZE, feature_map_h, feature_map_w, 64]

        output = deconv_bn_layer(input_data, [2, 2, 64, 128], deconv_output_shape, initializer=msra_initializer(),
                                 wd=cfg.WEIGHT_DECAY, loss_collection=self.loss_collection, use_bn=cfg.USE_BN,
                                 is_training=self.is_training, bn_decay=0.99, name='l17_deconv')

        # nonbt1d_18
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l18_nonbt1d')
        
        # nonbt1d_19
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l19_nonbt1d')

        # deconvolution 20(upsampling)
        feature_map_h, feature_map_w = get_deconv_output_feature_map_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 2)
        deconv_output_shape = [cfg.BATCH_SIZE, feature_map_h, feature_map_w, 32]

        output = deconv_bn_layer(output, [2, 2, 32, 64], deconv_output_shape, initializer=msra_initializer(),
                                 wd=cfg.WEIGHT_DECAY, loss_collection=self.loss_collection, use_bn=cfg.USE_BN,
                                 is_training=self.is_training, bn_decay=0.99, name='l20_deconv')

        # nonbt1d_21
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l21_nonbt1d')

        # nonbt1d_22
        output = res_nonbt1d(output, 3, initializer=msra_initializer(), wd=cfg.WEIGHT_DECAY, use_bn=cfg.USE_BN,
                             is_training=self.is_training, bn_decay=0.99, use_dropout=cfg.USE_DROPOUT,
                             drop_rate=cfg.DROP_RATE, loss_collection=self.loss_collection, name='l22_nonbt1d')

        # deconvolution 23(upsampling)
        feature_map_h, feature_map_w = get_deconv_output_feature_map_shape(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 1)
        deconv_output_shape = [cfg.BATCH_SIZE, feature_map_h, feature_map_w, cfg.NUM_CLASSES]

        classifier = deconv_bn_layer(output, [2, 2, cfg.NUM_CLASSES, 32], deconv_output_shape, initializer=msra_initializer(),
                                 wd=cfg.WEIGHT_DECAY, loss_collection=self.loss_collection, use_bn=cfg.USE_BN,
                                 is_training=self.is_training, bn_decay=0.99, name='l23_deconv')
        
        return classifier

    def add_forward_layer(self):
        if cfg.SUBTRACT_CHANNEL_MEAN:
            #red, green, blue shape = (batch_size, 1024, 2048, 1) (batch_size, 1024, 2048, 1) (batch_size, 1024, 2048, 1)
            red, green, blue = tf.split(self.input_image, 3, 3)
            # input_data shape = (batch_size, 1024, 2048, 3)
            input_data = tf.concat([
                red - BGR_MEAN[2],
                green - BGR_MEAN[1],
                blue - BGR_MEAN[0]
                ], 3)
        else:
            input_data = self.input_image
        
        forward_layer = self.build_encoder(input_data)
        self.segment_logit = self.build_segmenter(forward_layer)

        print('self.segment_logit = {}'.format(self.segment_logit.get_shape().as_list()))

    def trainables(self):
        trainables = tf.trainable_variables()
        return trainables
    
    def bn_variables(self):
        all_variables = tf.global_variables()
        bn_variables = []
        for var in all_variables:
            var_name = var.op.name
            var_basename = var_name.split('/')[-1]
            if 'bn' in var_name and ('moving_mean' == var_basename or 'moving_variance' == var_basename):
                bn_variables.append(var)

    def bn_mean_variance(self):
        default_graph = tf.get_default_graph()
        all_nodes = default_graph.as_graph_def().node
        bn_tensors = []
        for node in all_nodes:
            node_name = node.name
            node_basename = node.name.split('/')[-1]
            if 'bn' in node_name and ('mean' == node_basename or 'variance' == node_basename):
                tensor = default_graph.get_tensor_by_name(node_name + ':0')
                bn_tensors.append(tensor)
        return bn_tensors
